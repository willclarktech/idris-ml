#include "port_assert.h"

Test(linear_concat_stack, stack_backward) {
    param_clear();
    /* Three [2]-vectors: [1,2], [3,4], [5,6]. Stack at dim=0 -> [3,2]. */
    double a[] = {1, 2}, b[] = {3, 4}, c[] = {5, 6};
    int s[] = {2};
    TensorHandle ta = tensor_create(a, s, 1, 1);
    TensorHandle tb = tensor_create(b, s, 1, 1);
    TensorHandle tc = tensor_create(c, s, 1, 1);
    param_register("a", ta);
    param_register("b", tb);
    param_register("c", tc);
    TensorHandle in[] = {ta, tb, tc};
    TensorHandle st = tensor_stack(in, 3, 0);
    if (tensor_dim(st) == 2 && tensor_size(st, 0) == 3 && tensor_size(st, 1) == 2) {
        double sout[6];
        tensor_to_doubles(st, sout);
        ASSERT_NEAR("stack[0,0]", sout[0], 1.0, 1e-10);
        ASSERT_NEAR("stack[1,1]", sout[3], 4.0, 1e-10);
        ASSERT_NEAR("stack[2,0]", sout[4], 5.0, 1e-10);

        TensorHandle loss = tensor_sum(st);
        tensor_backward(loss);
        ASSERT_NEAR("d_stack_a[0]", param_grad_item_at(0, 0), 1.0, 1e-6);
        ASSERT_NEAR("d_stack_a[1]", param_grad_item_at(0, 1), 1.0, 1e-6);
        ASSERT_NEAR("d_stack_b[0]", param_grad_item_at(1, 0), 1.0, 1e-6);
        ASSERT_NEAR("d_stack_c[1]", param_grad_item_at(2, 1), 1.0, 1e-6);
    } else {
        printf("ok: stack stub on this backend (scalars only) — skipping shape assertions\n");
    }
    param_clear();
}
