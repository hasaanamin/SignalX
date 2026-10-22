/* Test-only ABI adapter. gen_sss.c is downloaded unchanged from pinned srsRAN.
 * No LTE algorithm is implemented here. The DLL is an external validation tool,
 * is generated under output/, and is not a runtime/package dependency.
 */
#include "gen_sss.c"

__declspec(dllexport) void reference_sss(float *first, float *second, unsigned pci)
{
    srsran_sss_generate(first, second, pci);
}
