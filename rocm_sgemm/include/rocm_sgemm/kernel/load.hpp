/*
 * MIT License
 *
 * Copyright (c) 2024 Adel Johar
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#ifndef ROCM_SGEMM_LOAD_HPP
#define ROCM_SGEMM_LOAD_HPP

namespace rocm_sgemm
{

/**
 * @brief OOB-clamped buffer load SRD configuration for gfx11/12.
 *
 * A raw buffer SRD with hardware out-of-bounds clamping: loads past the allocation return 0, so
 * ragged (unaligned) tiles need no per-element global bounds check.
 */
static constexpr unsigned buffer_rsrc_config = 0x31004000u;

/**
 * @brief Creates a 128-bit buffer resource descriptor with OOB clamping.
 *
 * @param ptr Base pointer for the buffer.
 * @param num_bytes Total allocation size in bytes for bounds checking.
 * @return 128-bit SRD for raw buffer instructions.
 */
static __device__ __forceinline__ __amdgpu_buffer_rsrc_t make_buffer_rsrc(const void* ptr,
                                                                          unsigned    num_bytes)
{
    return __builtin_amdgcn_make_buffer_rsrc(const_cast<void*>(ptr),
                                             0,
                                             num_bytes,
                                             static_cast<int>(buffer_rsrc_config));
}

/**
 * @brief Unified helper for vectorized buffer loads.
 *
 * @tparam V Vectorized data type (4/8/12/16 bytes).
 * @param rsrc Resource descriptor with clamped bounds.
 * @param byte_offset Flat byte offset from the buffer base.
 * @return The loaded vector (zero-padded if out of bounds).
 */
template<typename V>
static __device__ __forceinline__ V buffer_load(__amdgpu_buffer_rsrc_t rsrc, int byte_offset)
{
    if constexpr(sizeof(V) == 16)
    {
        using payload_t = float __attribute__((ext_vector_type(4)));
        payload_t raw   = __builtin_amdgcn_raw_buffer_load_b128(rsrc, byte_offset, 0, 0);
        V         result;
        __builtin_memcpy(&result, &raw, 16);
        return result;
    }
    else if constexpr(sizeof(V) == 12)
    {
        using payload_t = float __attribute__((ext_vector_type(3)));
        payload_t raw   = __builtin_amdgcn_raw_buffer_load_b96(rsrc, byte_offset, 0, 0);
        V         result;
        __builtin_memcpy(&result, &raw, 12);
        return result;
    }
    else if constexpr(sizeof(V) == 8)
    {
        using payload_t = float __attribute__((ext_vector_type(2)));
        payload_t raw   = __builtin_amdgcn_raw_buffer_load_b64(rsrc, byte_offset, 0, 0);
        V         result;
        __builtin_memcpy(&result, &raw, 8);
        return result;
    }
    else
    {
        static_assert(sizeof(V) == 4);
        float raw = __builtin_amdgcn_raw_buffer_load_b32(rsrc, byte_offset, 0, 0);
        V     result;
        __builtin_memcpy(&result, &raw, 4);
        return result;
    }
}

/**
 * @brief Byte-exact vector type selector.
 *
 * `ext_vector_type` rounds a vector's element count up to a power of two for size/alignment, so a
 * width-3 float vector has sizeof 16 (not 12). That padding makes register->LDS stores overwrite
 * the neighbouring cell. For power-of-two widths we keep the native vector type (single-instruction
 * stores); for the rest we use a plain array aggregate whose sizeof is exactly W * sizeof(B).
 */
template<class B, int W, bool IS_POW2 = ((W & (W - 1)) == 0)>
struct load_vector
{
    using type = B __attribute__((ext_vector_type(W)));
};

template<class B, int W>
struct load_vector<B, W, false>
{
    struct type
    {
        B data[W];
    };
};

/**
 * @brief Largest vector width (in elements) that divides both the contiguous extent and the tile.
 *
 * Picks the widest vector whose byte size fits within MAX_BITS while keeping every block-wide sweep
 * step aligned to the contiguous dimension.
 *
 * @tparam CONTIG Length of the contiguous axis the vector spans.
 * @tparam BLOCK_SIZE Number of threads in the block.
 * @tparam T Data type of the elements.
 * @tparam MAX_BITS Maximum bit-width for a single vectorized memory operation.
 * @return The chosen vector width in elements (>= 1).
 */
template<int CONTIG, int BLOCK_SIZE, class T, int MAX_BITS>
constexpr int load_vector_width()
{
    constexpr int max_vector_width = (MAX_BITS / 8) / static_cast<int>(sizeof(T));
    for(int w = max_vector_width; w > 1; --w)
    {
        if(CONTIG % w == 0 && (BLOCK_SIZE * w) % CONTIG == 0)
        {
            return w;
        }
    }
    return 1;
}

/**
 * @brief Prefetches a global-memory tile block into registers, vectorized on both sides.
 *
 * Two staging strategies are selected at compile time from the matrix layout:
 *
 * - Native (col-major A, row-major B): each thread reads `vector_width` consecutive elements along
 *   the LDS-contiguous m/n axis (coalesced) and commits them as one linear vector store.
 *
 * - Transposed (row-major A, col-major B): both sides stay vectorized by having each thread own
 *   ADJ_ROWS adjacent perp-rows. The read still grabs `vector_width` consecutive K at a fixed
 *   perp-row (coalesced); because the LDS tile is K-major, those adjacent rows are contiguous in
 *   LDS, so the commit gathers element j from the ADJ_ROWS registers into a width-ADJ_ROWS vector
 *   and emits one wide store per K-position. The transpose is realized by an in-register gather
 *   (VGPR moves), not a lane-by-lane scatter of narrow LDS stores.
 *
 * @tparam MATRIX Which matrix (A or B) is being loaded.
 * @tparam ACCESS Memory layout of the block being prefetched.
 * @tparam BLOCK_SIZE Number of threads participating in the prefetch.
 * @tparam BLOCK_M Number of rows in the block tile.
 * @tparam BLOCK_N Number of columns in the block tile.
 * @tparam PADDING Padding added to the LDS inner dimension to avoid bank conflicts.
 * @tparam T Data type of the elements.
 * @tparam MAX_BITS Maximum bit-width for vectorized memory operations.
 * @tparam ADJ_ROWS Adjacent perp-rows owned per thread in the transposed scheme (wide-store width).
 */
template<m_input  MATRIX,
         m_layout ACCESS,
         int      BLOCK_SIZE,
         int      BLOCK_M,
         int      BLOCK_N,
         int      PADDING,
         class T,
         int MAX_BITS = 128,
         int ADJ_ROWS = 2>
class prefetch_fragment
{
    static constexpr bool is_transposed
        = (MATRIX == m_input::matrix_a && ACCESS == m_layout::row_major)
          || (MATRIX == m_input::matrix_b && ACCESS == m_layout::col_major);

    // Perp axis = LDS-contiguous m/n; k axis = block_k. The LDS tile is stored [k][perp] with a
    // padded row stride.
    static constexpr int perp_dim  = (MATRIX == m_input::matrix_a) ? BLOCK_M : BLOCK_N;
    static constexpr int k_dim     = (MATRIX == m_input::matrix_a) ? BLOCK_N : BLOCK_M;
    static constexpr int lds_pitch = perp_dim + PADDING;

    // Vectorized global read width: native spans the perp axis, transposed spans the k axis.
    static constexpr int vector_width
        = is_transposed ? load_vector_width<k_dim, BLOCK_SIZE, T, MAX_BITS>()
                        : load_vector_width<perp_dim, BLOCK_SIZE, T, MAX_BITS>();

    // Native layout counts: running-cursor sweep over the perp-contiguous tile.
    static constexpr int total_vectors      = (BLOCK_M * BLOCK_N) / vector_width;
    static constexpr int vectors_per_thread = (total_vectors + BLOCK_SIZE - 1) / BLOCK_SIZE;
    static constexpr int guaranteed_iters   = total_vectors / BLOCK_SIZE;
    static constexpr int remainder_iters    = vectors_per_thread - guaranteed_iters;

    static constexpr int step_elems = BLOCK_SIZE * vector_width;
    static constexpr int iter_inc   = step_elems / perp_dim;
    static constexpr int off_inc    = step_elems % perp_dim;

    // Adjacent-row (transposed) counts: each thread owns ADJ_ROWS adjacent perp-rows per group.
    static constexpr int threads_per_k  = k_dim / vector_width;
    static constexpr int rows_per_block = BLOCK_SIZE / threads_per_k;
    static constexpr int vpt_groups
        = (perp_dim + ADJ_ROWS * rows_per_block - 1) / (ADJ_ROWS * rows_per_block);
    static constexpr int total_slots       = vpt_groups * ADJ_ROWS;
    static constexpr int guaranteed_groups = perp_dim / (ADJ_ROWS * rows_per_block);
    static constexpr int guaranteed_slots  = guaranteed_groups * ADJ_ROWS;
    static constexpr int remainder_slots   = total_slots - guaranteed_slots;

    // Row advance when the slot cursor crosses a group boundary (l wraps ADJ_ROWS-1 -> 0): the next
    // group's first row is ADJ_ROWS*rows_per_block ahead, minus the (ADJ_ROWS-1) unit steps
    // already taken within the group. Within a group the cursor advances one row per slot.
    static constexpr int group_row_step = ADJ_ROWS * rows_per_block - (ADJ_ROWS - 1);
    static constexpr int group_lds_step = ADJ_ROWS * rows_per_block;

    static constexpr int regs_size
        = (total_slots > vectors_per_thread) ? total_slots : vectors_per_thread;

    using base_type   = typename type_selector<T>::type;
    using vector_type = typename load_vector<base_type, vector_width>::type;

    vector_type regs[regs_size];

    __amdgpu_buffer_rsrc_t rsrc_;
    const T*               base_ptr_;

    __device__ __forceinline__ int lead_dim(int M, int N)
    {
        if constexpr(MATRIX == m_input::matrix_a)
        {
            return is_transposed ? N : M;
        }
        else
        {
            return is_transposed ? M : N;
        }
    }

    // Vector load at element index `idx` relative to the current tile pointer `input`, as a byte
    // offset from the SRD base. OOB indices clamp to zero in hardware.
    __device__ __forceinline__ vector_type load_at(const T* input, int idx) const
    {
        const int byte_off = static_cast<int>(reinterpret_cast<const char*>(input)
                                              - reinterpret_cast<const char*>(base_ptr_))
                             + idx * static_cast<int>(sizeof(T));
        return buffer_load<vector_type>(rsrc_, byte_off);
    }

    // Native running-cursor advance: step the perp-contiguous sweep by one block-wide stride, with
    // the fractional off_inc wrap. `stride` is `lead` for the global read, `lds_pitch` for the LDS
    // commit (perp_dim is the same contiguous extent in both spaces).
    __device__ __forceinline__ void
        advance_native(int stride, int perp_stride, int& iter, int& off, int& curr)
    {
        curr += iter_inc * stride;
        iter += iter_inc;
        if constexpr(off_inc != 0)
        {
            off += off_inc;
            curr += off_inc;
            if(off >= perp_dim)
            {
                off -= perp_dim;
                iter += 1;
                curr += stride - perp_stride;
            }
        }
    }

    __device__ __forceinline__ void advance_global_native(int lead, int& iter, int& off, int& curr)
    {
        advance_native(lead, perp_dim, iter, off, curr);
    }

    __device__ __forceinline__ void advance_lds_native(int& iter, int& off, int& curr)
    {
        advance_native(lds_pitch, perp_dim, iter, off, curr);
    }

public:
    /**
     * @brief Constructs the prefetch fragment and initializes the buffer SRD.
     *
     * @param base Base pointer to the global memory matrix.
     * @param alloc_elems Total elements in the global matrix for OOB clamping.
     */
    __device__ __forceinline__ prefetch_fragment(const T* base, unsigned alloc_elems)
        : rsrc_(make_buffer_rsrc(base, alloc_elems * static_cast<unsigned>(sizeof(T))))
        , base_ptr_(base)
    {}

    /**
     * @brief Prefetches a full tile from global memory into registers (native layout).
     *
     * Coalesced along the perp axis with a running-cursor sweep.
     *
     * @param input Pointer to the global memory tile offset.
     * @param M Global row count.
     * @param N Global column count.
     * @param tid Thread ID within the block.
     */
    template<bool TR = is_transposed>
    __device__ __forceinline__ auto prefetch(const T* input, int M, int N, int tid) ->
        typename std::enable_if<!TR, void>::type
    {
        const int lead     = lead_dim(M, N);
        const int base_idx = tid * vector_width;
        int       iter     = base_idx / perp_dim;
        int       off      = base_idx % perp_dim;
        int       curr     = iter * lead + off;

        auto fetch = [&]<size_t reg, bool checked>()
        {
            if constexpr(!checked)
            {
                regs[reg] = load_at(input, curr);
            }
            else
            {
                if(iter < k_dim)
                {
                    regs[reg] = load_at(input, curr);
                }
            }
            advance_global_native(lead, iter, off, curr);
        };

        if constexpr(guaranteed_iters > 0)
        {
            [&]<size_t... i>(std::index_sequence<i...>) {
                (fetch.template operator()<i, false>(), ...);
            }(std::make_index_sequence<guaranteed_iters>{});
        }
        if constexpr(remainder_iters > 0)
        {
            [&]<size_t... i>(std::index_sequence<i...>) {
                (fetch.template operator()<guaranteed_iters + i, true>(), ...);
            }(std::make_index_sequence<remainder_iters>{});
        }
    }

    /**
     * @brief Prefetches a full tile from global memory into registers (transposed layout).
     *
     * Each slot holds `vector_width` consecutive K at one perp-row; each thread owns ADJ_ROWS
     * adjacent perp-rows per group. The (row, curr) cursor accumulates: +1 row (+lead) within a
     * group, +group_row_step at a group boundary; base_k is a fixed per-thread offset folded into
     * the initial curr.
     *
     * @param input Pointer to the global memory tile offset.
     * @param M Global row count.
     * @param N Global column count.
     * @param tid Thread ID within the block.
     */
    template<bool TR = is_transposed>
    __device__ __forceinline__ auto prefetch(const T* input, int M, int N, int tid) ->
        typename std::enable_if<TR, void>::type
    {
        const int lead     = lead_dim(M, N);
        const int row_base = (tid / threads_per_k) * ADJ_ROWS;
        const int base_k   = (tid % threads_per_k) * vector_width;

        int row  = row_base;
        int curr = row_base * lead + base_k;

        auto fetch = [&]<size_t slot, bool checked>()
        {
            if constexpr(!checked)
            {
                regs[slot] = load_at(input, curr);
            }
            else
            {
                if(row < perp_dim)
                {
                    regs[slot] = load_at(input, curr);
                }
            }
            constexpr int l = static_cast<int>(slot) % ADJ_ROWS;
            if constexpr(l == ADJ_ROWS - 1)
            {
                row += group_row_step;
                curr += group_row_step * lead;
            }
            else
            {
                row += 1;
                curr += lead;
            }
        };

        if constexpr(guaranteed_slots > 0)
        {
            [&]<size_t... s>(std::index_sequence<s...>) {
                (fetch.template operator()<s, false>(), ...);
            }(std::make_index_sequence<static_cast<size_t>(guaranteed_slots)>{});
        }
        if constexpr(remainder_slots > 0)
        {
            [&]<size_t... s>(std::index_sequence<s...>) {
                (fetch.template operator()<guaranteed_slots + s, true>(), ...);
            }(std::make_index_sequence<static_cast<size_t>(remainder_slots)>{});
        }
    }

    /**
     * @brief Partially prefetches one pipeline slice of the native perp-contiguous sweep.
     *
     * @tparam STEP Current pipeline step index.
     * @tparam TOTAL_STEPS Total number of pipeline steps.
     * @param input Pointer to the global memory tile offset.
     * @param M Global row count.
     * @param N Global column count.
     * @param tid Thread ID within the block.
     */
    template<size_t STEP, size_t TOTAL_STEPS, bool TR = is_transposed>
    __device__ __forceinline__ auto partial_prefetch(const T* input, int M, int N, int tid) ->
        typename std::enable_if<!TR, void>::type
    {
        constexpr size_t vpt            = static_cast<size_t>(vectors_per_thread);
        constexpr size_t loads_per_step = (vpt + TOTAL_STEPS - 1) / TOTAL_STEPS;
        constexpr size_t start          = STEP * loads_per_step;
        constexpr size_t end   = (start + loads_per_step > vpt) ? vpt : start + loads_per_step;
        constexpr size_t count = (end > start) ? (end - start) : 0;

        if constexpr(count > 0)
        {
            const int lead     = lead_dim(M, N);
            const int base_idx = tid * vector_width;
            const int seed     = base_idx + static_cast<int>(start) * step_elems;
            int       iter     = seed / perp_dim;
            int       off      = seed % perp_dim;
            int       curr     = iter * lead + off;

            auto fetch = [&]<size_t reg, bool checked>()
            {
                if constexpr(!checked)
                {
                    regs[reg] = load_at(input, curr);
                }
                else
                {
                    if(iter < k_dim)
                    {
                        regs[reg] = load_at(input, curr);
                    }
                }
                advance_global_native(lead, iter, off, curr);
            };

            [&]<size_t... i>(std::index_sequence<i...>)
            {
                (
                    [&]()
                    {
                        constexpr size_t reg = start + i;
                        if constexpr(reg < static_cast<size_t>(guaranteed_iters))
                        {
                            fetch.template operator()<reg, false>();
                        }
                        else
                        {
                            fetch.template operator()<reg, true>();
                        }
                    }(),
                    ...);
            }(std::make_index_sequence<count>{});
        }
    }

    /**
     * @brief Partially prefetches one group-slice of the transposed adjacent-row sweep.
     *
     * The cursor is seeded to the group at g_start.
     *
     * @tparam STEP Current pipeline step index.
     * @tparam TOTAL_STEPS Total number of pipeline steps.
     * @param input Pointer to the global memory tile offset.
     * @param M Global row count.
     * @param N Global column count.
     * @param tid Thread ID within the block.
     */
    template<size_t STEP, size_t TOTAL_STEPS, bool TR = is_transposed>
    __device__ __forceinline__ auto partial_prefetch(const T* input, int M, int N, int tid) ->
        typename std::enable_if<TR, void>::type
    {
        constexpr int groups_per_step
            = (vpt_groups + static_cast<int>(TOTAL_STEPS) - 1) / static_cast<int>(TOTAL_STEPS);
        constexpr int g_start = static_cast<int>(STEP) * groups_per_step;
        constexpr int g_end
            = (g_start + groups_per_step > vpt_groups) ? vpt_groups : g_start + groups_per_step;
        constexpr int slot_start = g_start * ADJ_ROWS;
        constexpr int slot_count = (g_end > g_start) ? (g_end - g_start) * ADJ_ROWS : 0;

        if constexpr(slot_count > 0)
        {
            const int lead     = lead_dim(M, N);
            const int row_base = (tid / threads_per_k) * ADJ_ROWS;
            const int base_k   = (tid % threads_per_k) * vector_width;
            const int seed_row = row_base + g_start * group_lds_step;
            int       row      = seed_row;
            int       curr     = seed_row * lead + base_k;

            auto fetch = [&]<size_t slot, bool checked>()
            {
                if constexpr(!checked)
                {
                    regs[slot] = load_at(input, curr);
                }
                else
                {
                    if(row < perp_dim)
                    {
                        regs[slot] = load_at(input, curr);
                    }
                }
                constexpr int l = static_cast<int>(slot) % ADJ_ROWS;
                if constexpr(l == ADJ_ROWS - 1)
                {
                    row += group_row_step;
                    curr += group_row_step * lead;
                }
                else
                {
                    row += 1;
                    curr += lead;
                }
            };

            constexpr int step_guar_end = (slot_start + slot_count < guaranteed_slots)
                                              ? slot_start + slot_count
                                              : guaranteed_slots;
            constexpr int step_guar_count
                = (step_guar_end > slot_start) ? step_guar_end - slot_start : 0;
            constexpr int step_rem_start
                = (slot_start > guaranteed_slots) ? slot_start : guaranteed_slots;
            constexpr int step_rem_count = (slot_start + slot_count > step_rem_start)
                                               ? slot_start + slot_count - step_rem_start
                                               : 0;

            if constexpr(step_guar_count > 0)
            {
                [&]<size_t... s>(std::index_sequence<s...>) {
                    (fetch.template operator()<slot_start + s, false>(), ...);
                }(std::make_index_sequence<static_cast<size_t>(step_guar_count)>{});
            }
            if constexpr(step_rem_count > 0)
            {
                [&]<size_t... s>(std::index_sequence<s...>) {
                    (fetch.template operator()<step_rem_start + s, true>(), ...);
                }(std::make_index_sequence<static_cast<size_t>(step_rem_count)>{});
            }
        }
    }

    /**
     * @brief Commits prefetched register data to shared memory (native layout).
     *
     * Linear vector store with a running-cursor sweep.
     *
     * @param output Pointer to the LDS destination tile.
     * @param tid Thread ID within the block.
     */
    template<bool TR = is_transposed>
    __device__ __forceinline__ auto commit(T* output, int tid) ->
        typename std::enable_if<!TR, void>::type
    {
        const int base_idx = tid * vector_width;
        int       iter     = base_idx / perp_dim;
        int       off      = base_idx % perp_dim;
        int       curr     = iter * lds_pitch + off;

        auto store = [&]<size_t reg, bool checked>()
        {
            if constexpr(!checked)
            {
                *reinterpret_cast<vector_type*>(output + curr) = regs[reg];
            }
            else
            {
                if(iter < k_dim)
                {
                    *reinterpret_cast<vector_type*>(output + curr) = regs[reg];
                }
            }
            advance_lds_native(iter, off, curr);
        };

        if constexpr(guaranteed_iters > 0)
        {
            [&]<size_t... i>(std::index_sequence<i...>) {
                (store.template operator()<i, false>(), ...);
            }(std::make_index_sequence<guaranteed_iters>{});
        }
        if constexpr(remainder_iters > 0)
        {
            [&]<size_t... i>(std::index_sequence<i...>) {
                (store.template operator()<guaranteed_iters + i, true>(), ...);
            }(std::make_index_sequence<remainder_iters>{});
        }
    }

    /**
     * @brief Commits prefetched register data to shared memory (transposed layout).
     *
     * Adjacent-row wide store: gathers element j from the ADJ_ROWS registers into a width-ADJ_ROWS
     * vector and emits one store per K-position (adjacent rows are contiguous in the K-major LDS
     * tile). The LDS base cursor accumulates by group_lds_step per group. A partial trailing group
     * falls back to per-slot scalar stores of the `vector_width` K-values.
     *
     * @param output Pointer to the LDS destination tile.
     * @param tid Thread ID within the block.
     */
    template<bool TR = is_transposed>
    __device__ __forceinline__ auto commit(T* output, int tid) ->
        typename std::enable_if<TR, void>::type
    {
        using store_vec = base_type __attribute__((ext_vector_type(ADJ_ROWS)));

        const int row_base = (tid / threads_per_k) * ADJ_ROWS;
        const int base_k   = (tid % threads_per_k) * vector_width;
        const int bk_pitch = base_k * lds_pitch;

        int row_lo = row_base;

        // One group: gather element j from each of the ADJ_ROWS registers into a store_vec, then emit
        // one wide store into the K-major LDS tile (adjacent rows are contiguous). The store address
        // walks addr += lds_pitch across the vector_width K-positions from a per-group base.
        auto store_group = [&]<size_t g, bool checked>()
        {
            auto emit = [&]()
            {
                int  addr   = bk_pitch + row_lo;
                auto emit_j = [&]<size_t j>()
                {
                    store_vec v;
                    [&]<size_t... l>(std::index_sequence<l...>)
                    {
                        ((v[l] = reinterpret_cast<const base_type*>(
                              &regs[static_cast<int>(g) * ADJ_ROWS + static_cast<int>(l)])[j]),
                         ...);
                    }(std::make_index_sequence<static_cast<size_t>(ADJ_ROWS)>{});
                    *reinterpret_cast<store_vec*>(&output[addr]) = v;
                    addr += lds_pitch;
                };

                [&]<size_t... j>(std::index_sequence<j...>) {
                    (emit_j.template operator()<j>(), ...);
                }(std::make_index_sequence<static_cast<size_t>(vector_width)>{});
            };

            if constexpr(!checked)
            {
                emit();
            }
            else
            {
                if(row_lo + ADJ_ROWS - 1 < perp_dim)
                {
                    emit();
                }
            }
            row_lo += group_lds_step;
        };

        if constexpr(guaranteed_groups > 0)
        {
            [&]<size_t... g>(std::index_sequence<g...>) {
                (store_group.template operator()<g, false>(), ...);
            }(std::make_index_sequence<static_cast<size_t>(guaranteed_groups)>{});
        }

        // Remainder: partial group, fall back to per-slot scalar stores of the vector_width K-values.
        if constexpr(remainder_slots > 0)
        {
            [&]<size_t... s>(std::index_sequence<s...>)
            {
                (
                    [&]()
                    {
                        constexpr int slot = guaranteed_slots + static_cast<int>(s);
                        constexpr int g    = slot / ADJ_ROWS;
                        constexpr int l    = slot % ADJ_ROWS;
                        const int     row  = row_base + g * group_lds_step + l;
                        if(row < perp_dim)
                        {
                            const base_type* src  = reinterpret_cast<const base_type*>(&regs[slot]);
                            int              addr = bk_pitch + row;
                            [&]<size_t... j>(std::index_sequence<j...>) {
                                ((output[addr + static_cast<int>(j) * lds_pitch] = src[j]), ...);
                            }(std::make_index_sequence<static_cast<size_t>(vector_width)>{});
                        }
                    }(),
                    ...);
            }(std::make_index_sequence<static_cast<size_t>(remainder_slots)>{});
        }
    }
};

} // namespace rocm_sgemm

#endif // ROCM_SGEMM_LOAD_HPP
