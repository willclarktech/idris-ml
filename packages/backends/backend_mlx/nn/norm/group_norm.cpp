/* tensor_group_norm for the mlx backend.
 *
 * mlx has no native group_norm. Implement on host via the same per-
 * group mean/var/normalize loop as tape — the result is non-grad
 * because mlx's autograd surface doesn't trace through this code
 * path, but no current example calls group_norm in a grad context.
 * Inputs are staged as double regardless of underlying dtype so the
 * loop stays dtype-agnostic. */
#include "../../tensor.h"
#include "../../stream.h"
#include "../../precision.h"
#include <cmath>
#include <cstdlib>
#include <vector>

extern "C" TensorHandle tensor_group_norm(TensorHandle hinput, TensorHandle hgamma,
                                          TensorHandle hbeta, int numGroups, int channels,
                                          int spatial, double eps) {
	auto inp = (Tensor*)hinput;
	auto gamma = (Tensor*)hgamma;
	auto beta = (Tensor*)hbeta;
	int n = channels * spatial;
	int chPerGroup = channels / numGroups;
	int groupSize = chPerGroup * spatial;
	mx::eval(inp->data);
	mx::eval(gamma->data);
	mx::eval(beta->data);
	std::vector<double> inpD_buf((size_t)n);
	std::vector<double> gammaD_buf((size_t)channels);
	std::vector<double> betaD_buf((size_t)channels);
	mx_to_doubles(inp->data, inpD_buf.data());
	mx_to_doubles(gamma->data, gammaD_buf.data());
	mx_to_doubles(beta->data, betaD_buf.data());
	const double* inpD = inpD_buf.data();
	const double* gammaD = gammaD_buf.data();
	const double* betaD = betaD_buf.data();
	double* out = (double*)calloc(n, sizeof(double));
	for (int g = 0; g < numGroups; g++) {
		double mean = 0;
		int base = g * groupSize;
		for (int j = 0; j < groupSize; j++)
			mean += inpD[base + j];
		mean /= groupSize;
		double var = 0;
		for (int j = 0; j < groupSize; j++) {
			double d = inpD[base + j] - mean;
			var += d * d;
		}
		var /= groupSize;
		double rstd = 1.0 / sqrt(var + eps);
		for (int c = 0; c < chPerGroup; c++) {
			int absC = g * chPerGroup + c;
			for (int s = 0; s < spatial; s++) {
				int idx = absC * spatial + s;
				double x_hat = (inpD[idx] - mean) * rstd;
				out[idx] = gammaD[absC] * x_hat + betaD[absC];
			}
		}
	}
	auto result = mx_array_from_doubles(out, {n}, inp->data.dtype());
	free(out);
	return (TensorHandle)(new Tensor(result, inp->requires_grad || gamma->requires_grad));
}
