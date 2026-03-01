/**
 * @file backend_auto_detect.c
 * @brief Implementation of automatic DLL/SO export scanning and vtable population.
 *
 * See backend_auto_detect.h for usage documentation.
 *
 * Three public entry points:
 *   fb_classify_symbol()         — classify one symbol name into (op_id, conv)
 *   fb_enumerate_and_populate()  — scan all DLL/SO exports, fill ext_ops[op][conv]
 *   fb_auto_populate_ext_ops()   — fill from a caller-supplied symbol table
 *   fb_auto_populate_fn_ops()    — fill FB_CONV_CBLAS slots from fn-ptr table
 */

/* ── Platform DLL symbol lookup ─────────────────────────────────────────── */
#ifdef _WIN32
#  ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN
#  endif
#  include <windows.h>
#  include <winnt.h>      /* IMAGE_EXPORT_DIRECTORY, IMAGE_NT_HEADERS, etc. */
#  define FB_LOAD_SYM(handle, sym_name) \
       ((void *)GetProcAddress((HMODULE)(handle), (sym_name)))
#elif defined(__APPLE__)
#  include <dlfcn.h>
#  include <mach-o/dyld.h>
#  include <mach-o/loader.h>
#  include <mach-o/nlist.h>
#  define FB_LOAD_SYM(handle, sym_name) dlsym((handle), (sym_name))
#else
   /* Linux / BSD */
#  include <dlfcn.h>
#  include <link.h>      /* dl_iterate_phdr                                 */
#  include <elf.h>
#  define FB_LOAD_SYM(handle, sym_name) dlsym((handle), (sym_name))
#endif

#include <string.h>
#include <stdlib.h>      /* bsearch                                         */

#include "../backends/backend_auto_detect.h" /* fb_sym_entry_t, fb_fn_entry_t  */
#include "../judge/judge_op_ids.h"           /* FB_JUDGE_MAX_OPERATIONS       */

/* ── Reverse stem → FB_OP_* table (binary-searched) ────────────────────── */

/* Forward-declared here; defined in op_vtable_map.c alongside the forward
   table.  fb_stem_to_op_id() performs a binary search over k_op_stem_map[].*/
extern uint32_t fb_stem_to_op_id(const char *stem);

/* ── fb_classify_symbol ─────────────────────────────────────────────────── */

/**
 * Classify a single exported symbol name into a (op_id, conv) pair.
 *
 * Recognised patterns:
 *   "cblas_<stem>"  → FB_CONV_CBLAS
 *   "<stem>_"       → FB_CONV_FORTRAN  (exact trailing underscore)
 *   "<stem>_ref"    → FB_CONV_REF
 *
 * Returns FB_CONV_COUNT (== 3) if the symbol cannot be matched.
 */
fb_conv_t fb_classify_symbol(const char *name, uint32_t *out_op_id)
{
    if (!name || !out_op_id) return FB_CONV_COUNT;

    size_t len = strlen(name);

    /* ---- CBLAS: "cblas_<stem>" ----------------------------------------- */
    if (len > 6 && memcmp(name, "cblas_", 6) == 0) {
        const char *stem = name + 6;
        uint32_t id = fb_stem_to_op_id(stem);
        if (id < (uint32_t)FB_JUDGE_MAX_OPERATIONS) {
            *out_op_id = id;
            return FB_CONV_CBLAS;
        }
        return FB_CONV_COUNT;
    }

    /* ---- REF: "<stem>_ref" ---------------------------------------------- */
    if (len > 4 && memcmp(name + len - 4, "_ref", 4) == 0) {
        /* Copy stem (without trailing "_ref") to a stack buffer.
           BLAS/LAPACK stem names are at most ~20 chars.                     */
        if (len - 4 < 64) {
            char stem[64];
            memcpy(stem, name, len - 4);
            stem[len - 4] = '\0';
            uint32_t id = fb_stem_to_op_id(stem);
            if (id < (uint32_t)FB_JUDGE_MAX_OPERATIONS) {
                *out_op_id = id;
                return FB_CONV_REF;
            }
        }
        return FB_CONV_COUNT;
    }

    /* ---- FORTRAN: "<stem>_" (single trailing underscore, not "_ref") ---- */
    if (len > 1 && name[len - 1] == '_') {
        if (len - 1 < 64) {
            char stem[64];
            memcpy(stem, name, len - 1);
            stem[len - 1] = '\0';
            uint32_t id = fb_stem_to_op_id(stem);
            if (id < (uint32_t)FB_JUDGE_MAX_OPERATIONS) {
                *out_op_id = id;
                return FB_CONV_FORTRAN;
            }
        }
        return FB_CONV_COUNT;
    }

    return FB_CONV_COUNT;
}

/* ── Helper: fill one ext_ops slot if empty ─────────────────────────────── */

static void fill_slot(fb_backend_vtable_t *vtable, const char *name,
                      void *lib_handle)
{
    uint32_t op_id;
    fb_conv_t conv = fb_classify_symbol(name, &op_id);
    if (conv == FB_CONV_COUNT) return;  /* not a recognised BLAS/LAPACK name */
    if (vtable->ext_ops[op_id][conv] != NULL) return;  /* already filled    */

    void *fn = FB_LOAD_SYM(lib_handle, name);
    if (fn) {
        vtable->ext_ops[op_id][conv] = (fb_generic_fn)fn;
    }
}

/* ── Platform export scanners ───────────────────────────────────────────── */

#ifdef _WIN32
/* Walk the PE export directory of a loaded module.
   This reads the in-memory image, which is always available for a loaded DLL. */
static void scan_pe_exports(fb_backend_vtable_t *vtable, void *lib_handle)
{
    HMODULE mod = (HMODULE)lib_handle;

    /* Locate the PE header.  The DOS stub at the module base contains the
       offset to IMAGE_NT_HEADERS.                                            */
    BYTE *base = (BYTE *)mod;
    IMAGE_DOS_HEADER   *dos  = (IMAGE_DOS_HEADER *)base;
    if (dos->e_magic != IMAGE_DOS_SIGNATURE) return;

    IMAGE_NT_HEADERS   *nt   = (IMAGE_NT_HEADERS *)(base + dos->e_lfanew);
    if (nt->Signature != IMAGE_NT_SIGNATURE) return;

    IMAGE_DATA_DIRECTORY *dir =
        &nt->OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_EXPORT];
    if (dir->VirtualAddress == 0 || dir->Size == 0) return;

    IMAGE_EXPORT_DIRECTORY *exp =
        (IMAGE_EXPORT_DIRECTORY *)(base + dir->VirtualAddress);

    DWORD  count    = exp->NumberOfNames;
    DWORD *rva_names = (DWORD *)(base + exp->AddressOfNames);

    for (DWORD i = 0; i < count; i++) {
        const char *sym_name = (const char *)(base + rva_names[i]);
        fill_slot(vtable, sym_name, lib_handle);
    }
}

#elif defined(__APPLE__)
/* Walk the Mach-O symbol table of a loaded dylib. */
static void scan_macho_exports(fb_backend_vtable_t *vtable, void *lib_handle)
{
    /* dladdr gives us the image base for any symbol; we use a known symbol.  */
    Dl_info info;
    if (dladdr((void *)fb_classify_symbol, &info) == 0) return;

    /* Iterate all loaded images; find the one matching lib_handle.           */
    uint32_t img_count = _dyld_image_count();
    for (uint32_t img = 0; img < img_count; img++) {
        const struct mach_header *hdr = _dyld_get_image_header(img);
        intptr_t slide               = _dyld_get_image_vmaddr_slide(img);
        if (!hdr) continue;

        /* Walk load commands looking for LC_SYMTAB. */
        const struct load_command *lc =
            (const struct load_command *)((const char *)hdr +
                                          sizeof(struct mach_header_64));
        const struct symtab_command *symtab = NULL;
        for (uint32_t cmd = 0; cmd < hdr->ncmds; cmd++) {
            if (lc->cmd == LC_SYMTAB) { symtab = (const struct symtab_command *)lc; break; }
            lc = (const struct load_command *)((const char *)lc + lc->cmdsize);
        }
        if (!symtab) continue;

        /* Find the linkedit segment to compute symbol/string table addresses.*/
        lc = (const struct load_command *)((const char *)hdr + sizeof(struct mach_header_64));
        const struct segment_command_64 *linkedit = NULL;
        for (uint32_t cmd = 0; cmd < hdr->ncmds; cmd++) {
            if (lc->cmd == LC_SEGMENT_64) {
                const struct segment_command_64 *seg = (const struct segment_command_64 *)lc;
                if (strcmp(seg->segname, "__LINKEDIT") == 0) { linkedit = seg; break; }
            }
            lc = (const struct load_command *)((const char *)lc + lc->cmdsize);
        }
        if (!linkedit) continue;

        uintptr_t linkedit_base = (uintptr_t)slide + linkedit->vmaddr - linkedit->fileoff;
        const struct nlist_64 *syms =
            (const struct nlist_64 *)(linkedit_base + symtab->symoff);
        const char *strtab = (const char *)(linkedit_base + symtab->stroff);

        for (uint32_t s = 0; s < symtab->nsyms; s++) {
            if ((syms[s].n_type & N_TYPE) != N_SECT) continue;
            if (syms[s].n_un.n_strx == 0) continue;
            const char *sym_name = strtab + syms[s].n_un.n_strx;
            /* Mach-O exported symbols have a leading '_'. */
            if (sym_name[0] == '_') sym_name++;
            fill_slot(vtable, sym_name, lib_handle);
        }
    }
    (void)info; /* suppress unused-variable warning */
}

#else  /* Linux / BSD */

struct scan_ctx { fb_backend_vtable_t *vtable; void *lib_handle; };

static int scan_phdr_cb(struct dl_phdr_info *info, size_t size, void *data)
{
    struct scan_ctx *ctx = (struct scan_ctx *)data;
    (void)size;

    /* Walk PT_DYNAMIC to find DT_SYMTAB, DT_STRTAB, DT_SYMENT, DT_HASH. */
    const ElfW(Phdr) *phdr = info->dlpi_phdr;
    for (ElfW(Half) i = 0; i < info->dlpi_phnum; i++, phdr++) {
        if (phdr->p_type != PT_DYNAMIC) continue;

        const ElfW(Dyn) *dyn  = (const ElfW(Dyn) *)(info->dlpi_addr + phdr->p_vaddr);
        ElfW(Sym)      *symtab = NULL;
        const char     *strtab = NULL;
        ElfW(Word)     *hash   = NULL;
        size_t          syment = sizeof(ElfW(Sym));

        for (const ElfW(Dyn) *d = dyn; d->d_tag != DT_NULL; d++) {
            switch (d->d_tag) {
            case DT_SYMTAB: symtab = (ElfW(Sym)  *)d->d_un.d_ptr; break;
            case DT_STRTAB: strtab = (const char *)d->d_un.d_ptr; break;
            case DT_SYMENT: syment = (size_t)d->d_un.d_val;       break;
            case DT_HASH:   hash   = (ElfW(Word) *)d->d_un.d_ptr; break;
            default: break;
            }
        }
        if (!symtab || !strtab || !hash) continue;

        /* DT_HASH: [nbuckets, nchain, buckets[nbuckets], chains[nchain]]
           nchain == number of dynsym entries.                               */
        ElfW(Word) nchain = hash[1];
        for (ElfW(Word) s = 0; s < nchain; s++) {
            ElfW(Sym) *sym = (ElfW(Sym) *)((char *)symtab + s * syment);
            if (ELF64_ST_TYPE(sym->st_info) != STT_FUNC) continue;
            if (sym->st_name == 0) continue;
            const char *sym_name = strtab + sym->st_name;
            fill_slot(ctx->vtable, sym_name, ctx->lib_handle);
        }
        break; /* Only one PT_DYNAMIC segment per image. */
    }
    return 0;  /* continue iterating */
}
#endif /* platform */

/* ── fb_enumerate_and_populate ──────────────────────────────────────────── */

void fb_enumerate_and_populate(fb_backend_vtable_t *vtable, void *lib_handle)
{
    if (!vtable || !lib_handle) return;

#ifdef _WIN32
    scan_pe_exports(vtable, lib_handle);
#elif defined(__APPLE__)
    scan_macho_exports(vtable, lib_handle);
#else
    struct scan_ctx ctx = { vtable, lib_handle };
    dl_iterate_phdr(scan_phdr_cb, &ctx);
#endif
}

/* ── fb_auto_populate_ext_ops ───────────────────────────────────────────── */

void fb_auto_populate_ext_ops(fb_backend_vtable_t  *vtable,
                               void                 *lib_handle,
                               const fb_sym_entry_t *table,
                               size_t                count)
{
    if (!vtable || !lib_handle || !table || count == 0) return;

    for (size_t i = 0; i < count; i++) {
        uint32_t  id   = table[i].op_id;
        fb_conv_t conv = table[i].conv;

        /* Bounds checks — protect against stale generated tables. */
        if (id   >= (uint32_t)FB_JUDGE_MAX_OPERATIONS) continue;
        if ((int)conv < 0 || conv >= FB_CONV_COUNT)    continue;

        /* Never overwrite a slot already filled. */
        if (vtable->ext_ops[id][conv] != NULL) continue;

        void *fn = FB_LOAD_SYM(lib_handle, table[i].symbol);
        if (fn) {
            vtable->ext_ops[id][conv] = (fb_generic_fn)fn;
        }
    }
}

/* ── fb_auto_populate_fn_ops ────────────────────────────────────────────── */

void fb_auto_populate_fn_ops(fb_backend_vtable_t *vtable,
                              const fb_fn_entry_t *table,
                              size_t               count)
{
    if (!vtable || !table || count == 0) return;

    for (size_t i = 0; i < count; i++) {
        uint32_t id = table[i].op_id;

        if (id >= (uint32_t)FB_JUDGE_MAX_OPERATIONS) continue;
        /* Compile-linked backends are always CBLAS-convention. */
        if (vtable->ext_ops[id][FB_CONV_CBLAS] != NULL) continue;
        if (table[i].fn == NULL) continue;

        vtable->ext_ops[id][FB_CONV_CBLAS] = table[i].fn;
    }
}
