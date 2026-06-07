/* probes.c — see probes.h for the interface contract. */

#include "probes.h"
#include <string.h>
#include <sys/utsname.h>

#ifndef _WIN32
#include <sys/stat.h>
#include <dlfcn.h>
#endif

int idrisml_probe_os(void) {
	struct utsname u;
	if (uname(&u) != 0) return 3;
	if (strcmp(u.sysname, "Darwin") == 0) return 0;
	if (strcmp(u.sysname, "Linux") == 0) return 1;
	if (strstr(u.sysname, "MINGW") || strstr(u.sysname, "MSYS") || strstr(u.sysname, "Windows"))
		return 2;
	return 3;
}

int idrisml_probe_arch(void) {
	struct utsname u;
	if (uname(&u) != 0) return 2;
	if (strcmp(u.machine, "arm64") == 0 || strcmp(u.machine, "aarch64") == 0) return 0;
	if (strcmp(u.machine, "x86_64") == 0 || strcmp(u.machine, "amd64") == 0) return 1;
	return 2;
}

int idrisml_probe_metal_available(void) {
#ifdef __APPLE__
	struct stat st;
	return stat("/System/Library/Frameworks/Metal.framework", &st) == 0 ? 1 : 0;
#else
	return 0;
#endif
}

int idrisml_probe_cuda_available(void) {
#if defined(__APPLE__) || defined(_WIN32)
	return 0;
#else
	void* h = dlopen("libcuda.so.1", RTLD_LAZY | RTLD_LOCAL);
	if (h) {
		dlclose(h);
		return 1;
	}
	return 0;
#endif
}
