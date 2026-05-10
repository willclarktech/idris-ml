/* Phase 6 modular tree placeholder for backend_mlx/.
   The backend's implementation currently lives in
   ../backend_mlx.cpp (the shrinking monolith); per-op extractions
   land alongside this stub during Phase 6a-6f and inherit the per-TU
   compile rule wired up in the Makefile. Symbol kept anchored so the
   stub TU contributes something concrete to the link — otherwise empty
   .cpp files can elide from some archive workflows. */
extern "C" int _backend_mlx_modular_present(void) { return 1; }
