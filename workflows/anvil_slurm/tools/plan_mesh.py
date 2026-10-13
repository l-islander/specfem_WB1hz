#!/usr/bin/env python3
"""plan_mesh.py -- size the 1.5 s / 1 s mesh for Anvil from the MEASURED 2 s mesh.

Every number is a ratio against the stage-2 mesh that actually ran on the 5090
(work/forward/result_M6/*/output_solver.txt, 2026-08):

    NEX 104 x 208 x 25 = 540,800 elements, NGLL 5
    element size  min 1197 m (y) / max 1291 m (topography-stretched z)
    minimum period resolved 1.966 s   (SPECFEM's own estimate, M0 model)
    DT 0.0145 s = the mesher's maximum suggested step (Courant 0.485)
    NSTEP 6900 -> 100 s records

Minimum period scales with the LARGEST element / Vs_min; the stable DT with
the SMALLEST element. The domain is fixed (131.09 x 248.99 km, 30 km deep,
topography to +1.9 km). So for a target period T:

    h_target = 1291 m * (T / 1.966)      -> NEX = ceil(L / h)  (then rounded UP
                                            to a multiple of the decomposition)
    DT       = 0.0145 * h_min / 1197     -> NSTEP = ceil(100 s / DT)

Decomposition: Anvil nodes have 128 cores, so NPROC must be a multiple of 128
and NEX_XI / NEX_ETA multiples of NPROC_XI / NPROC_ETA. The planner lists every
(NPROC_XI x NPROC_ETA) in {128,256,512,1024,2048} ranks that fits, with
elements/rank, and picks the one closest to --target-per-rank (default 2000:
the CPU solver's efficient range is ~1000-4000 elements per rank).

Cost model (refine with 4_bench.sh): --core-rate = EFFECTIVE M element-
timesteps per second per core, wall-clock based (MPI + I/O included). Default
0.15: the 2026-09-18 dry run on the 5090 box's CPU (16 ranks, 480k elements,
6897 steps) took 24 min per forward = 0.14; SPECFEM3D is memory-bandwidth
bound, so Milan at 128 ranks/node should land in 0.1-0.3 -- MEASURE IT (the
5090 does ~64 on the same mesh). The same dry run measured the SAVE_FORWARD
leg at 1.0x a plain forward and the kernel leg at 3.4x (80 vs 24 min), so one
iteration = NEV forwards + NEV x (1.0 save + 3.4 kernel) + NLS x NSUB trial
forwards = 930 forward-equivalents at 150/3/40. Anvil SU = core-hour.

Usage:
    tools/plan_mesh.py 1.5                    # table of options for 1.5 s
    tools/plan_mesh.py 1.0 --core-rate 1.5    # cost with a different rate
    tools/plan_mesh.py 1.5 --apply            # write mesh_plan.sh + DATA/{Mesh_Par_file,interfaces.dat,Par_file}
    tools/plan_mesh.py 1.5 --apply --nproc-xi 16 --nproc-eta 32
"""
import argparse, math, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
REF = dict(nex=(104, 208, 25), hmin=1197.0, hmax=1291.0, tmin=1.966, dt=0.0145, nstep=6900,
           elsteps=104 * 208 * 25 * 6900)
LX, LY, DEPTH, TOPO = 131090.0, 248990.0, 30000.0, 1900.0
RECORD_S = 100.0
CORES = 128
NPROC_CHOICES = [128, 256, 512, 1024, 2048]


def factor_pairs(n):
    return [(a, n // a) for a in range(1, n + 1) if n % a == 0 and a <= n // a * 4 and n // a <= a * 4]


def ceil_to(x, m):
    return int(math.ceil(x / m) * m)


def plan(T, nproc_xi=None, nproc_eta=None, target_per_rank=2000, core_rate=0.15,
         nev=150, nls=3, nsub=40, throttle=4, cores=CORES):
    global NPROC_CHOICES
    NPROC_CHOICES = [cores * k for k in (1, 2, 4, 8, 16)]
    h = REF["hmax"] * T / REF["tmin"]
    nx0, ny0, nz0 = LX / h, LY / h, (DEPTH + TOPO) / h
    opts = []
    for np_ in NPROC_CHOICES:
        for (pxi, peta) in factor_pairs(np_):
            if nproc_xi and (pxi, peta) != (nproc_xi, nproc_eta):
                continue
            nx, ny = ceil_to(nx0, pxi), ceil_to(ny0, peta)
            nz = int(math.ceil(nz0))
            nspec = nx * ny * nz
            per_rank = nspec / np_
            opts.append(dict(np=np_, pxi=pxi, peta=peta, nx=nx, ny=ny, nz=nz, nspec=nspec, per_rank=per_rank))
    if not opts:
        sys.exit("no decomposition matches")
    for o in opts:
        hx, hy, hz = LX / o["nx"], LY / o["ny"], DEPTH / o["nz"]
        o["hmin"], o["hmax"] = min(hx, hy, hz), max(hx, hy, hz * (DEPTH + TOPO) / DEPTH)
        o["tmin"] = REF["tmin"] * o["hmax"] / REF["hmax"]
        dt = REF["dt"] * o["hmin"] / REF["hmin"]
        o["dt"] = math.floor(dt * 1e4) / 1e4
        o["nstep"] = int(math.ceil(RECORD_S / o["dt"]))
        o["elsteps"] = o["nspec"] * o["nstep"]
        faces = 2 * o["nx"] * o["nz"] + 2 * o["ny"] * o["nz"] + o["nx"] * o["ny"]
        o["wave_gb"] = faces * 25 * 3 * 4 * o["nstep"] / 1e9
        o["kern_gb"] = o["nspec"] * 125 * 4 * 4 / 1e9
        o["model_gb"] = o["nspec"] * 125 * 4 * 3 / 1e9
        o["lbfgs_gb"] = o["nspec"] * 125 * 8 * 2 * (2 * 5 + 6) / 1e9
        o["fwd_s"] = o["elsteps"] / (core_rate * 1e6 * o["np"])
        fe = nev * (1 + 1.0 + 3.4) + nls * nsub
        o["iter_nodeh"] = fe * o["elsteps"] / (core_rate * 1e6 * cores) / 3600
        o["iter_su"] = o["iter_nodeh"] * cores
        o["nodes"] = o["np"] // cores
        o["iter_wall_h"] = o["iter_nodeh"] / (o["nodes"] * throttle)
        o["band"] = "HX" if o["dt"] <= 0.0125 else "BX"
    opts.sort(key=lambda o: (abs(math.log(o["per_rank"] / target_per_rank)), o["np"]))
    return opts


def show(T, opts, core_rate, throttle):
    best = opts[0]
    print(f"target period {T} s  (h_max <= {REF['hmax'] * T / REF['tmin']:.0f} m; 2 s reference: {REF['elsteps']/1e9:.2f} G el-steps/forward)")
    print(f"assumed core rate {core_rate} M el-steps/s/core, {throttle} events in flight  -- MEASURE with 4_bench.sh\n")
    print(f"{'NPROC':>5} {'XIxETA':>7} {'NEX':>12} {'NSPEC':>9} {'el/rank':>7} {'Tmin':>5} {'DT':>7} {'NSTEP':>6} "
          f"{'x2s':>5} {'fwd~':>6} {'wave/ev':>8} {'kern/ev':>8} {'model':>6} {'iterSU':>7} {'iterWall':>8}")
    for o in opts[:12]:
        mark = " <-" if o is best else ""
        print(f"{o['np']:5d} {o['pxi']:>3}x{o['peta']:<3} {o['nx']:>3}x{o['ny']:<3}x{o['nz']:<3} {o['nspec']:9d} {o['per_rank']:7.0f} "
              f"{o['tmin']:5.2f} {o['dt']:7.4f} {o['nstep']:6d} {o['elsteps']/REF['elsteps']:5.1f} {o['fwd_s']/60:5.1f}m "
              f"{o['wave_gb']:6.0f}GB {o['kern_gb']:6.1f}GB {o['model_gb']:5.1f}G {o['iter_su']:7.0f} {o['iter_wall_h']:6.1f}h{mark}")
    print(f"\nchosen: NPROC={best['np']} ({best['pxi']}x{best['peta']}, {best['nodes']} nodes/event), NEX {best['nx']}x{best['ny']}x{best['nz']}, "
          f"DT={best['dt']} ({best['band']}), NSTEP={best['nstep']}")
    print(f"  transient scratch at {throttle} events in flight: {best['wave_gb'] * throttle:.0f} GB;  kernels for 150 events: {best['kern_gb'] * 150:.0f} GB (x2 with clipped copies)")
    print(f"  L-BFGS update RAM ~{best['lbfgs_gb']:.0f} GB -> {'highmem partition' if best['lbfgs_gb'] > 200 else 'fits a 256 GB node'}")
    print(f"  per-node solver memory ~{best['nspec'] / best['nodes'] * 16e3 / 1e9:.0f} GB (kernel leg, 16 KB/element) of 256")


def apply(T, o):
    mpf_src = os.path.join(ROOT, "..", "DATA", "meshfem3D_files", "Mesh_Par_file")
    if not os.path.exists(mpf_src):
        mpf_src = os.path.join(ROOT, "DATA", "meshfem3D_files", "Mesh_Par_file.template")
    s = open(mpf_src).read()
    def setkey(txt, k, v):
        return re.sub(rf"(?m)^(\s*{k}\s*)=.*$", f"{k:<32}= {v}", txt)
    for k, v in (("NEX_XI", o["nx"]), ("NEX_ETA", o["ny"]), ("NPROC_XI", o["pxi"]), ("NPROC_ETA", o["peta"])):
        s = setkey(s, k, v)
    s = re.sub(r"(?m)^\s*1\s+\d+\s+1\s+\d+\s+1\s+\d+\s+-1\s*$",
               f"1              {o['nx']}           1               {o['ny']}             1        {o['nz']}        -1", s)
    dd = os.path.join(ROOT, "DATA", "meshfem3D_files"); os.makedirs(dd, exist_ok=True)
    open(os.path.join(dd, "Mesh_Par_file"), "w").write(s)
    itf = ("# number of interfaces\n 1\n# SUPPRESS_UTM_PROJECTION  NXI  NETA LONG_MIN   LAT_MIN    SPACING_XI SPACING_ETA\n"
           " .false. 354 468 -123.678855d0 43.642567d0 0.005d0 0.005d0\n interface2.dat\n"
           "# for each layer, we give the number of spectral elements in the vertical direction\n"
           f" {o['nz']}\n")
    open(os.path.join(dd, "interfaces.dat"), "w").write(itf)
    par = os.path.join(ROOT, "DATA", "Par_file"); p = open(par).read()
    for k, v in (("NPROC", o["np"]), ("NSTEP", o["nstep"]), ("DT", o["dt"]), ("GPU_MODE", ".false.")):
        p = setkey(p, k, v)
    open(par, "w").write(p)
    band = "1_3" if T <= 1.25 else "1.5_4"
    sig = 1000 if T <= 1.25 else 1500
    open(os.path.join(ROOT, "mesh_plan.sh"), "w").write(
        f"# written by tools/plan_mesh.py {T} --apply  -- sourced by config.sh; do not edit by hand\n"
        f"MESH_TARGET_S={T}\nNEX_XI={o['nx']}; NEX_ETA={o['ny']}; NEX_Z={o['nz']}\n"
        f"NPROC_XI={o['pxi']}; NPROC_ETA={o['peta']}; NPROC={o['np']}\nDT={o['dt']}; NSTEP={o['nstep']}\n"
        f"WAVEFIELD_GB_EST={int(o['wave_gb']) + 1}\nCALADJ_BANDS_PLAN={band},2_6,3_10,6_18\nSMOOTH_SIGMA_H_PLAN={sig}\n"
        f"# expected: Tmin~{o['tmin']:.2f} s, band code {o['band']}, {o['nodes']} nodes/event, iteration ~{o['iter_su']:.0f} SU at the planner's core rate\n")
    print(f"\nwrote {dd}/Mesh_Par_file, interfaces.dat, {par}, {ROOT}/mesh_plan.sh")
    print("next: ./2_make_mesh.sh  (re-mesh is REQUIRED after any plan change; models do not transfer across meshes)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("period", type=float, help="target minimum period (s), e.g. 1.5 or 1.0")
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--nproc-xi", type=int); ap.add_argument("--nproc-eta", type=int)
    ap.add_argument("--target-per-rank", type=int, default=2000)
    ap.add_argument("--core-rate", type=float, default=0.15, help="effective M element-timesteps/s per core, wall based (4_bench.sh measures it)")
    ap.add_argument("--throttle", type=int, default=4, help="events in flight (ARRAY_THROTTLE)")
    ap.add_argument("--cores-per-node", type=int, default=CORES, help="128 on Anvil; 16 for a mock run on the 5090 box")
    a = ap.parse_args()
    if bool(a.nproc_xi) != bool(a.nproc_eta):
        sys.exit("give both --nproc-xi and --nproc-eta")
    opts = plan(a.period, a.nproc_xi, a.nproc_eta, a.target_per_rank, a.core_rate, throttle=a.throttle, cores=a.cores_per_node)
    show(a.period, opts, a.core_rate, a.throttle)
    if a.apply:
        apply(a.period, opts[0])


if __name__ == "__main__":
    main()
