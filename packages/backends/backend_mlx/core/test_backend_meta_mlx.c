/* mlx backend_name accessor (core/backend_meta.cpp). No mlx test invoked it. */
#include <criterion/criterion.h>
#include "backend.h"

#ifdef BACKEND_MLX

Test(backend_meta_mlx, name_is_mlx) {
	cr_assert_str_eq(backend_name(), "mlx", "backend_name should be \"mlx\" (got %s)",
	                 backend_name());
}

#endif /* BACKEND_MLX */
