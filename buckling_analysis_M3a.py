#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Buckling Analysis — M3 (updated)
--------------------------------
Adds FULL (Koiter load–deflection + contour) for --core m3,
and respects --core in SENS/SOBOL.
- --core m3 : two-term Ritz + root rotational spring (default)
- --core m2 : one-term (fast, clamped-like)
- --mode full  : single case (plots + Results + DeflectionGrid)
- --mode sens  : OAT sensitivity (SENS)
- --mode sobol : Saltelli-lite Sobol (UQ / UQ_Control)
"""

import argparse, json, warnings
from math import pi
from pathlib import Path
import numpy as np
import pandas as pd

# --------- Matplotlib backend (robust fallback) ---------
import matplotlib
try:
    # user's preference for interactive backends; fallback to Agg if unavailable
    matplotlib.use('TkAgg')
except Exception:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ---------------- Micromech & CLT ----------------
def micromech_ud(Ef, Em, Gf, Gm, nuf, num, Vf):
    Vm = 1.0 - Vf
    E1 = Vf*Ef + Vm*Em
    rE = Ef/Em if Em>0 else 1e9; etaE=(rE-1)/(rE+2.0)
    E2 = Em*(1+2*etaE*Vf)/(1-etaE*Vf)
    rG = Gf/Gm if Gm>0 else 1e9; etaG=(rG-1)/(rG+1.0)
    G12= Gm*(1+2*etaG*Vf)/(1-etaG*Vf)
    nu12 = Vf*nuf + Vm*num
    return float(E1), float(E2), float(G12), float(nu12)

def Q_reduced(E1, E2, G12, nu12):
    nu21 = nu12 * E2 / max(E1,1e-18)
    denom = max(1.0 - nu12*nu21, 1e-18)
    return E1/denom, E2/denom, nu12*E2/denom, G12

def Qbar_from_Q(Q11, Q22, Q12, Q66, th_deg):
    th=np.deg2rad(th_deg); m=np.cos(th); n=np.sin(th); m2=m*m; n2=n*n
    Q11b = Q11*m2*m2 + 2*(Q12+2*Q66)*m2*n2 + Q22*n2*n2
    Q22b = Q11*n2*n2 + 2*(Q12+2*Q66)*m2*n2 + Q22*m2*m2
    Q12b = (Q11+Q22-4*Q66)*m2*n2 + Q12*(m2*m2+n2*n2)
    Q16b = (Q11 - Q12 - 2*Q66)*m2*m*n - (Q22 - Q12 - 2*Q66)*n2*n*m
    Q26b = (Q11 - Q12 - 2*Q66)*m*n2*m - (Q22 - Q12 - 2*Q66)*n*m2*n
    Q66b = (Q11 + Q22 - 2*Q12 - 2*Q66)*m2*n2 + Q66*(m2*m2+n2*n2)
    return np.array([[Q11b,Q12b,Q16b],[Q12b,Q22b,Q26b],[Q16b,Q26b,Q66b]], dtype=float)

def ABD_from_layup(E1,E2,G12,nu12, angles, t_total):
    n=max(len(angles),1); tply=t_total/n
    Q11,Q22,Q12,Q66 = Q_reduced(E1,E2,G12,nu12)
    z=[-t_total/2.0]
    for _ in range(n): z.append(z[-1]+tply)
    A=np.zeros((3,3)); B=np.zeros((3,3)); D=np.zeros((3,3))
    for k in range(n):
        th = angles[k] if k < len(angles) else 0.0
        Qb=Qbar_from_Q(Q11,Q22,Q12,Q66, th)
        zk1,zk=z[k],z[k+1]
        A+=Qb*(zk-zk1); B+=0.5*Qb*(zk**2 - zk1**2); D+=(1.0/3.0)*Qb*(zk**3 - zk1**3)
    return A,B,D


# ---------------- Geometry & weights ----------------
def build_geometry(vals, PPW=60, nx_min=1801):
    L=float(vals["L"]); b_r=float(vals["b_root"]); b_t=float(vals["b_tip"])
    h_r=float(vals["h_root"]); h_t=float(vals["h_tip"]); t_f=float(vals["t_face_total"])
    beta_guess = 1.5*pi/max(b_r,1e-18); lam_guess = 2*pi/max(beta_guess,1e-9)
    nx = max(int(nx_min), int(np.ceil((L/lam_guess)*PPW)))
    x = np.linspace(0.0, L, nx)
    b = b_r + (b_t - b_r)*(x/max(L,1e-18))
    h = h_r + (h_t - h_r)*(x/max(L,1e-18)); H = h + 2*t_f
    ky = pi/np.maximum(b,1e-18)
    return L,x,b,h,H,ky,b_r

# ---------------- Basis ----------------
def basis_terms(a,beta,x):
    expax=np.exp(-a*x); sin1=np.sin(beta*x); cos1=np.cos(beta*x)
    F1  = expax*(sin1);             F1p = expax*(beta*cos1 - a*sin1)
    F1pp= expax*(-beta**2*sin1 - 2*a*beta*cos1 + a**2*sin1)
    F2  = expax*(x*sin1);           F2p = expax*(sin1 + x*beta*cos1 - a*x*sin1)
    F2pp= expax*(2*beta*cos1 - x*beta**2*sin1 - 2*a*(sin1 + x*beta*cos1) + a**2*x*sin1)
    return (F1,F1p,F1pp,F2,F2p,F2pp)

# ---------------- M3 matrices (N, D) ----------------
def build_mats_factory_m3(x,b,h,H,ky, A_f,D_f,D_w, alpha0, b_root, Ktheta_root_per_m):
    Df11,Df22,Df12,Df66 = float(D_f[0,0]),float(D_f[1,1]),float(D_f[0,1]),float(D_f[2,2])
    Dw22 = float(D_w[1,1])
    int_s2   = (b/2.0); int_sy2 = ( (pi**2)/(2.0*np.maximum(b,1e-18)) ); int_syy2 = (b/2.0)*(ky**4)
    ktheta_web_pair = 2.0*(4.0*Dw22/np.maximum(h,1e-12))

    def N_pair(Fi,Fpi,Fppi, Fj,Fpj,Fppj):
        Ub = 0.5*np.trapz( Df11*(Fppi*Fppj)*int_s2
                          +2.0*(Df12+2.0*Df66)*(-ky**2)*(Fppi*Fj)*(b/2.0)
                          +Df22*(Fi*Fj)*int_syy2
                          +4.0*Df66*(Fpi*Fpj)*int_sy2, x)
        Ue = np.trapz( ktheta_web_pair*(Fi*Fj)*(ky**2), x);  return Ub+Ue
    def D_pair(Fpi,Fpj): return 0.5*np.trapz( alpha0*(Fpi*Fpj)*(b/2.0), x )

    def build_ND(a,beta):
        F1,F1p,F1pp,F2,F2p,F2pp = basis_terms(a,beta,x)
        N11 = N_pair(F1,F1p,F1pp, F1,F1p,F1pp) + 0.5*Ktheta_root_per_m*(beta**2)*(b_root/2.0)
        N12 = N_pair(F1,F1p,F1pp, F2,F2p,F2pp); N21=N12
        N22 = N_pair(F2,F2p,F2pp, F2,F2p,F2pp)
        D11 = D_pair(F1p,F1p); D12=D_pair(F1p,F2p); D21=D12; D22=D_pair(F2p,F2p)
        N=np.array([[N11,N12],[N21,N22]],float); D=np.array([[D11,D12],[D21,D22]],float)
        return N, D, (F1,F1p,F1pp,F2,F2p,F2pp)
    return build_ND

# ---------------- α–β finder ----------------
def fast_ab(Ktheta_root_per_m, b_root, k_y_root, build_ND, kappa_scale, return_vec=False):
    # κ-based windows
    kappa = (Ktheta_root_per_m / max(kappa_scale, 1e-18)) if kappa_scale>0 else 10.0
    if kappa<=0.1: aL,aU,bL,bU=8.0,24.0,0.5*k_y_root,2.2*k_y_root
    elif kappa<=1.0: aL,aU,bL,bU=10.0,26.0,0.7*k_y_root,2.2*k_y_root
    else: aL,aU,bL,bU=12.0,28.0,0.9*k_y_root,2.2*k_y_root

    def eigmin(a,b):
        N,D,_=build_ND(a,b)
        try:
            M = np.linalg.solve(D, N)
            lam, vec = np.linalg.eig(M)
            lam = np.real(lam); vec = np.real(vec)
            j = int(np.argmin(lam))
            return lam[j], vec[:,j]
        except Exception:
            return np.inf, np.array([np.nan,np.nan])

    A=np.linspace(aL,aU,9); B=np.linspace(bL,bU,33)
    bestP=np.inf; bestA=bestB=None; bestv=None
    for a in A:
        vals=[eigmin(a,b) for b in B]
        arr = np.array([v[0] for v in vals])
        j=int(np.argmin(arr))
        if 0<arr[j]<bestP:
            bestP=arr[j]; bestA=a; bestB=B[j]; bestv = vals[j][1]
    # refine (few iterations)
    for _ in range(8):
        Bs=np.linspace(max(1e-6,bestB*0.88),bestB*1.12,9)
        vals=[eigmin(bestA,b) for b in Bs]
        arr=np.array([v[0] for v in vals]); jj=int(np.argmin(arr))
        bestB=Bs[jj]; bestP=arr[jj]; bestv = vals[jj][1]
        As=np.linspace(max(1e-3,bestA*0.88),bestA*1.12,9)
        vals=[eigmin(a,bestB) for a in As]
        arr=np.array([v[0] for v in vals]); jj=int(np.argmin(arr))
        bestA=As[jj]; bestP=arr[jj]; bestv = vals[jj][1]
    if return_vec:
        return float(bestP), float(bestA), float(bestB), (bestv if bestv is not None else np.array([1.0,0.0]))
    return float(bestP), float(bestA), float(bestB)

# ---------------- M2 (fast) Pcr only ----------------
def eval_m2_Pcr(vals, PPW=30, nx_min=801):
    L,x,b,h,H,ky,b_root = build_geometry(vals, PPW=PPW, nx_min=nx_min)
    Ef,Em,Gf,num,nuf,Vf = float(vals["Ef"]),float(vals["Em"]),float(vals["Gf"]),float(vals["num"]),float(vals["nuf"]),float(vals["Vf"])
    face_angles=[float(z) for z in str(vals["face_angles"]).split(",")] if str(vals.get("face_angles","")).strip()!="" else [0.0]
    web_angles =[float(z) for z in str(vals["web_angles"]).split(",")]  if str(vals.get("web_angles","")).strip()!=""  else [0.0]
    t_f=float(vals["t_face_total"]); t_w=float(vals["t_web_total"])
    # CLT
    E1,E2,G12,nu12 = micromech_ud(Ef,Em,Gf,Em/(2*(1+num)),nuf,num,Vf)
    A_f,B_f,D_f = ABD_from_layup(E1,E2,G12,nu12, face_angles, t_f)
    A_w,B_w,D_w = ABD_from_layup(E1,E2,G12,nu12, web_angles,  t_w)
    Df11,Df22,Df12,Df66 = float(D_f[0,0]),float(D_f[1,1]),float(D_f[0,1]),float(D_f[2,2])
    Dw22 = float(D_w[1,1])
    # mapping
    w_f=float(vals["w_f"]); t_fl=t_f+t_w
    I_faces_panel=2.0*(b*t_f)*(H/2.0)**2; I_faces_flange=2.0*(w_f*t_fl)*(H/2.0)**2; I_webs=2.0*(t_w*h**3)/12.0
    I_total=I_faces_panel+I_faces_flange+I_webs; alpha0=t_f*(H/2.0)/np.maximum(I_total,1e-18)*(L-x)
    # weights
    int_s2=(b/2.0); int_sy2=(pi**2)/(2.0*np.maximum(b,1e-18)); int_syy2=(b/2.0)*(ky**4)
    ktheta_web_pair=2.0*(4.0*Dw22/np.maximum(h,1e-12))
    def P_of(a,beta):
        expax=np.exp(-a*x); sinbx=np.sin(beta*x); cosbx=np.cos(beta*x)
        F=expax*(x*sinbx); Fp=expax*(sinbx + x*beta*cosbx - a*x*sinbx)
        Fpp=expax*(2*beta*cosbx - x*beta**2*sinbx - 2*a*(sinbx + x*beta*cosbx) + a**2*x*sinbx)
        Ub=0.5*np.trapz(Df11*(Fpp**2)*int_s2 + 2*(Df12+2*Df66)*(-ky**2)*(Fpp*F)*(b/2.0) + Df22*(F**2)*int_syy2 + 4*Df66*(Fp**2)*int_sy2, x)
        Ue=np.trapz(ktheta_web_pair*(F**2)*(ky**2), x); Den=0.5*np.trapz(alpha0*(Fp**2)*int_s2, x)
        return (Ub+Ue)/max(Den,1e-18)
    beta_guess=1.5*pi/max(b_root,1e-18); aL,aU=10.0,26.0; bL,bU=0.7*(pi/max(b_root,1e-18)),2.2*(pi/max(b_root,1e-18))
    A=np.linspace(aL,aU,7); B=np.linspace(bL,bU,19)
    Pbest=np.inf
    for a in A:
        valsb=[P_of(a,beta) for beta in B]; m=min(valsb)
        if 0<m<Pbest: Pbest=m
    return float(Pbest)

# ---------------- M3 Pcr + eigenvector ----------------
def eval_m3_Pcr_and_mode(vals, PPW=60, nx_min=1801, return_mode=True):
    # Geometry
    L,x,b,h,H,ky,b_root = build_geometry(vals, PPW=PPW, nx_min=nx_min)
    # Materials
    Ef,Em,Gf,num,nuf,Vf = float(vals["Ef"]),float(vals["Em"]),float(vals["Gf"]),float(vals["num"]),float(vals["nuf"]),float(vals["Vf"])
    face_angles=[float(z) for z in str(vals["face_angles"]).split(",")] if str(vals.get("face_angles","")).strip()!="" else [0.0]
    web_angles =[float(z) for z in str(vals["web_angles"]).split(",")]  if str(vals.get("web_angles","")).strip()!=""  else [0.0]
    t_f=float(vals["t_face_total"]); t_w=float(vals["t_web_total"])
    Gm=Em/(2*(1+num))
    E1,E2,G12,nu12 = micromech_ud(Ef,Em,Gf,Gm,nuf,num,Vf)
    A_f,B_f,D_f = ABD_from_layup(E1,E2,G12,nu12, face_angles, t_f)
    A_w,B_w,D_w = ABD_from_layup(E1,E2,G12,nu12, web_angles,  t_w)

    # Mapping
    w_f=float(vals["w_f"]); t_fl=t_f+t_w
    I_faces_panel=2.0*(b*t_f)*(H/2.0)**2; I_faces_flange=2.0*(w_f*t_fl)*(H/2.0)**2; I_webs=2.0*(t_w*h**3)/12.0
    I_total=I_faces_panel+I_faces_flange+I_webs
    alpha0 = t_f*(H/2.0)/np.maximum(I_total,1e-18)*(L-x)

    # Root spring (if absent, assume large — clamped-like)
    Ktheta = float(vals.get("Ktheta_root_per_m", 1e9))

    # Build N,D factory
    build_ND = build_mats_factory_m3(x,b,h,H,ky, A_f,D_f,D_w, alpha0, b_root, Ktheta)
    # κ-scale uses D11 ~ bending stiffness scale; and β*_clamped ~ pi/b_root
    kappa_scale = float(D_f[0,0]) * max(pi/max(b_root,1e-18), 1e-12)

    # Find a*, b*, and (optionally) eigenvector
    Pcr, a_star, b_star, vec = fast_ab(Ktheta, b_root, pi/max(b_root,1e-18), build_ND, kappa_scale, return_vec=True)
    if not return_mode:
        return float(Pcr)

    # Recover F(x) and F'(x) using eigenvector
    N,D,(F1,F1p,F1pp,F2,F2p,F2pp) = build_ND(a_star, b_star)
    c1, c2 = (vec[0], vec[1])
    # Normalize mode shape (avoid arbitrary scaling)
    norm = max(np.sqrt(np.trapz((c1*F1p + c2*F2p)**2, x)), 1e-18)
    c1 /= norm; c2 /= norm
    F   = c1*F1 + c2*F2
    Fp  = c1*F1p + c2*F2p
    # useful returns
    out = {
        "Pcr": float(Pcr),
        "alpha_star": float(a_star),
        "beta_star": float(b_star),
        "lambda_x": float(2.0*pi/max(b_star,1e-18)),
        "x": x, "b": b, "h": h, "H": H, "ky": ky,
        "F": F, "Fp": Fp,
        "A_f": A_f, "A_w": A_w, "D_f": D_f, "D_w": D_w,
        "alpha0": alpha0, "w_f": w_f, "L": L
    }
    return out

# ---------------- Koiter & curves (general, uses mode fields) ----------------
def koiter_curves_from_mode(mode):
    x = mode["x"]; b=mode["b"]; H=mode["H"]; ky=mode["ky"]
    F=mode["F"]; Fp=mode["Fp"]; L=mode["L"]
    A_f=mode["A_f"]; A_w=mode["A_w"]; alpha0=mode["alpha0"]; w_f=mode["w_f"]
    Pcr=mode["Pcr"]
    # Den2
    Den2 = np.trapz(alpha0*(Fp**2)*(b/2.0), x)
    # k4 integrals
    A11=float(A_f[0,0]); A22=float(A_f[1,1]); A12=float(A_f[0,1]); A66=float(A_f[2,2])
    I_phi4    = 3.0*b/8.0
    I_phi2py2 = (ky**2)*b/4.0
    I_py4     = 3.0/8.0 * (ky**4) * b
    k4_int = 0.5 * ( (A11*(0.25)*(Fp**4)*I_phi4) +
                     (2*A12*(0.25)*(Fp**2)*(F**2)*I_phi2py2) +
                     (A22*(0.25)*F**4*I_py4) +
                     (2*A66*(F**2)*(Fp**2)*I_phi2py2) )
    k4 = float(np.trapz(k4_int, x))

    # Linear compliance via equivalent EI
    EI_faces=A11*b*(H**2)/2.0; EI_webs=float(A_w[0,0])*(H**3)/6.0; EI_flange=2.0*(A11+float(A_w[0,0]))*w_f*(H**2)/2.0
    EI_total=EI_faces+EI_webs+EI_flange
    fb = np.trapz((L-x)*x/np.maximum(EI_total,1e-18), x)

    # Curves up to 1.5 Pcr
    P = np.linspace(0.0, 1.5*Pcr, 181)
    a = np.zeros_like(P); idx = P > Pcr
    if k4>1e-18 and Den2>1e-18:
        a[idx] = np.sqrt( ((P[idx]-Pcr) * Den2) / k4 )
    theta_fac = np.sqrt( (1.0/L) * np.trapz( (F**2)*(ky**2), x) )
    delta_lin = P*fb
    delta_loc = L*theta_fac*a
    delta_tot = delta_lin + delta_loc
    dcr = float(np.interp(Pcr, P, delta_tot))
    return P, delta_lin, delta_loc, delta_tot, dcr, dict(Den2=Den2, k4=k4, fb=fb)

# ---------------- Plot helpers ----------------
def plot_load_deflection(case_label, Pcr, dcr, P, dlin, dloc, dtot, out_png):
    dmax=1.5*dcr; Pmax=1.5*Pcr
    # Build a clean linear reference
    fb_fit = float(np.dot(P, dlin)/max(np.dot(P,P),1e-18)) if P[-1]>0 else float(dlin[-1]/max(P[-1],1e-18))
    d_lin_full = np.linspace(0.0, dmax, 400); P_lin_full = d_lin_full / max(fb_fit,1e-18)

    # Clip by axes
    def clip(P_arr, d_arr):
        m=(P_arr<=Pmax) & (d_arr<=dmax+1e-15)
        return d_arr[m], P_arr[m]
    dloc_c,P_loc_c = clip(P, dloc); dtot_c,P_tot_c=clip(P, dtot)
    d_at_Pcr = float(np.interp(Pcr, P_tot_c if len(P_tot_c)>2 else P, dtot_c if len(P_tot_c)>2 else dtot))

    plt.figure(figsize=(7.6,4.8))
    plt.plot(d_lin_full, P_lin_full, '--', linewidth=2.0, label='linear only')
    plt.plot(dloc_c, P_loc_c, ':', linewidth=2.2, alpha=0.95, label='nonlinear only')
    plt.plot(dtot_c, P_tot_c, '-', linewidth=2.4, label='total = linear + nonlinear')
    plt.scatter([d_at_Pcr],[Pcr], s=70, edgecolor='k', zorder=4, label=f'Pcr≈{Pcr:.0f} N')
    plt.xlabel("Tip deflection δ [m]"); plt.ylabel("Load P [N]")
    plt.xlim(0,dmax); plt.ylim(0,Pmax)
    plt.grid(True, linestyle='--', alpha=0.3); plt.legend(loc='lower right', framealpha=0.9)
    plt.title(f"Load–deflection — {case_label}")
    plt.tight_layout(); plt.savefig(out_png, dpi=180); plt.close()

def save_deflection_grid_and_contour(x, b, F, case_label, xls_path, grid="100x50"):
    # Build grid
    try:
        nx, ny = [int(s) for s in str(grid).lower().replace('x',' ').split()]
    except Exception:
        nx, ny = 100, 50
    X = np.linspace(x[0], x[-1], nx)
    # for each X, local width is interpolated from b(x)
    bX = np.interp(X, x, b)
    Y = np.array([np.linspace(0.0, bX[i], ny) for i in range(nx)])  # (nx, ny)
    FX = np.interp(X, x, F)  # (nx,)
    # w(x,y)=F(x)*sin(pi*y/b(x))
    W = np.zeros((nx, ny))
    for i in range(nx):
        ky_i = pi/max(bX[i],1e-18)
        W[i,:] = float(FX[i]) * np.sin(ky_i * Y[i,:])
    # normalize for plotting
    maxabs = max(np.max(np.abs(W)), 1e-18)
    Wn = W / maxabs

    # Save contour png
    out_png = Path(xls_path).with_name("Contour.png")
    plt.figure(figsize=(7.2,3.8))
    # build a rectangular grid for imshow: x by y normalized to [0,1] in width
    extent=[X[0], X[-1], 0.0, 1.0]
    # map y to normalized (0..1) per-section for display only
    Yn = np.array([Y[i,:]/max(bX[i],1e-18) for i in range(nx)])
    plt.imshow(Wn.T, origin='lower', aspect='auto', extent=extent, interpolation='bilinear')
    plt.colorbar(label='w / max|w|')
    plt.xlabel('x [m]'); plt.ylabel('y/b(x) [-]')
    plt.title(f"Mode contour — {case_label}")
    plt.tight_layout(); plt.savefig(out_png, dpi=180); plt.close()

    # Save DeflectionGrid sheet
    rows=[]
    for i in range(nx):
        for j in range(ny):
            rows.append({"i":i, "j":j, "x [m]": float(X[i]), "y [m]": float(Y[i,j]), "w_norm": float(Wn[i,j])})
    df = pd.DataFrame(rows)
    with pd.ExcelWriter(xls_path, engine="openpyxl", mode="a", if_sheet_exists="overlay") as writer:
        df.to_excel(writer, sheet_name="DeflectionGrid", index=False)
    return str(out_png)

# ---------------- FULL (m2/m3) ----------------
def run_full(xls_path, core="m3", grid="100x50"):
    xls=Path(xls_path)
    df=pd.read_excel(xls, sheet_name="Inputs")
    vals={r["Key"]: r["Value"] for _,r in df.iterrows()}
    case_label = str(vals.get("case_name","CASE"))
    if "core" in vals: core = str(vals["core"]).lower()

    if core=="m2":
        Pcr=eval_m2_Pcr(vals, PPW=int(vals.get("PPW",30)), nx_min=int(vals.get("nx_min",801)))
        with pd.ExcelWriter(xls, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
            pd.DataFrame([{"P_cr [N]":Pcr, "core":"m2 (one-term clamped-fast)"}]).to_excel(writer, sheet_name="Results", index=False)
        return {"P_cr":Pcr, "core":"m2"}

    # ----- m3 FULL: compute mode + Koiter + plots + sheets -----
    mode = eval_m3_Pcr_and_mode(vals, PPW=int(vals.get("PPW",60)), nx_min=int(vals.get("nx_min",1801)), return_mode=True)
    P, dlin, dloc, dtot, dcr, more = koiter_curves_from_mode(mode)
    out_png_ld = xls.with_name("Full_LoadDeflection.png")
    plot_load_deflection(case_label, mode["Pcr"], dcr, P, dlin, dloc, dtot, out_png_ld)

    # Save Contour and Grid
    out_png_contour = save_deflection_grid_and_contour(mode["x"], mode["b"], mode["F"], case_label, xls, grid=grid)

    # Results sheet
    res = [{
        "P_cr [N]": mode["Pcr"],
        "delta_cr [m]": dcr,
        "alpha* [1/m]": mode["alpha_star"],
        "beta* [1/m]": mode["beta_star"],
        "lambda_x* [m]": mode["lambda_x"],
        "core": "m3 (two-term + root spring)",
        "Ktheta_root [N·m/m]": float(vals.get("Ktheta_root_per_m", 1e9))
    }]
    with pd.ExcelWriter(xls, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
        pd.DataFrame(res).to_excel(writer, sheet_name="Results", index=False)

    return {"P_cr": mode["Pcr"], "delta_cr": dcr, "load_deflection_plot": str(out_png_ld), "contour_plot": out_png_contour, "core":"m3"}

# ---------------- SENS (OAT) ----------------
def run_sens(xls_path, core="m2"):
    xls=Path(xls_path)
    df_inputs=pd.read_excel(xls, sheet_name="Inputs"); vals0={r["Key"]: r["Value"] for _,r in df_inputs.iterrows()}
    if "core" in vals0: core=str(vals0["core"]).lower()
    df_sens=pd.read_excel(xls, sheet_name="SENS")
    rows=[]
    for _,r in df_sens.iterrows():
        if int(r.get("enable",1))!=1: continue
        name=str(r["name"]); mode=str(r.get("mode","percent")).lower(); n=int(r.get("n_points",5))
        if mode=="percent":
            pct=float(r.get("delta_percent",0.1)); v0=float(vals0[name]); lo=v0*(1-pct); hi=v0*(1+pct)
        else:
            lo=float(r.get("low", float(vals0[name]))); hi=float(r.get("high", float(vals0[name])))
        grid=np.linspace(lo,hi,n)
        for v in grid:
            vals=vals0.copy(); vals[name]=float(v)
            if core=="m3":
                Pcr = eval_m3_Pcr_and_mode(vals, return_mode=False)
            else:
                Pcr = eval_m2_Pcr(vals)
            rows.append({"name":name,"value":v,"P_cr [N]":float(Pcr)})
    df_out = pd.DataFrame(rows)
    with pd.ExcelWriter(xls.with_name("M3_SENS_Results.xlsx")) as writer:
        df_out.to_excel(writer, sheet_name="SENS_Results", index=False)

    # Baseline
    if core=="m3":
        P0 = eval_m3_Pcr_and_mode(vals0, return_mode=False)
    else:
        P0 = eval_m2_Pcr(vals0)
    with pd.ExcelWriter(xls.with_name("M3_SENS_Results.xlsx"), mode="a", if_sheet_exists="replace") as writer:
        pd.DataFrame([{"baseline_P_cr [N]": float(P0), "core": core}]).to_excel(writer, sheet_name="SENS_Baseline", index=False)
    return {"rows":len(rows), "core":core}

# ---------------- SOBOL ----------------
def run_sobol(xls_path, core="m2"):
    xls=Path(xls_path)
    df_inputs=pd.read_excel(xls, sheet_name="Inputs"); vals={r["Key"]: r["Value"] for _,r in df_inputs.iterrows()}
    if "core" in vals: core=str(vals["core"]).lower()
    df_uq=pd.read_excel(xls, sheet_name="UQ")
    if "enable" in df_uq.columns: df_uq=df_uq[df_uq["enable"]==1].reset_index(drop=True)
    df_ctrl=pd.read_excel(xls, sheet_name="UQ_Control")
    N_base=int(df_ctrl.loc[df_ctrl["key"]=="N_base","value"].values[0])
    seed_col = df_ctrl.loc[df_ctrl["key"]=="seed","value"]
    seed=int(seed_col.values[0]) if len(seed_col)>0 else 1234

    names=list(df_uq["name"].astype(str).values); k=len(names)
    rng=np.random.default_rng(seed)
    def sample(low,high,N): return low+(high-low)*rng.random(N)
    lows=[float(df_uq.iloc[j].get("low",0.0)) for j in range(k)]
    highs=[float(df_uq.iloc[j].get("high",1.0)) for j in range(k)]
    A=np.vstack([sample(lows[j],highs[j],N_base) for j in range(k)]).T
    B=np.vstack([sample(lows[j],highs[j],N_base) for j in range(k)]).T

    def eval_row(row):
        v=vals.copy()
        for j,name in enumerate(names): v[name]=float(row[j])
        if core=="m3":
            return float(eval_m3_Pcr_and_mode(v, return_mode=False))
        else:
            return float(eval_m2_Pcr(v))

    YA=np.array([eval_row(A[n,:]) for n in range(N_base)])
    YB=np.array([eval_row(B[n,:]) for n in range(N_base)])
    # AB_i matrices
    YAB_list=[]
    for i in range(k):
        M=A.copy(); M[:,i]=B[:,i]
        YAB=np.array([eval_row(M[n,:]) for n in range(N_base)]); YAB_list.append(YAB)

    Y_all=np.concatenate([YA,YB]); V=np.var(Y_all, ddof=1) if len(Y_all)>1 else 1.0
    S=np.zeros(k); ST=np.zeros(k)
    for i in range(k):
        YAB=YAB_list[i]
        S[i]=np.mean(YB*(YAB-YA))/V if V>0 else 0.0
        ST[i]=0.5*np.mean((YA-YAB)**2)/V if V>0 else 0.0

    order=np.argsort(-ST); names_sorted=[names[i] for i in order]; S_sorted=S[order]; ST_sorted=ST[order]
    fig_path = xls.with_name(f"M3_Sobol_bars_core_{core}.png")
    plt.figure(figsize=(7.8,4.6)); xloc=np.arange(len(names_sorted))
    plt.bar(xloc-0.18, S_sorted, 0.36, label="S_i"); plt.bar(xloc+0.18, ST_sorted, 0.36, label="S_Ti")
    plt.xticks(xloc, names_sorted, rotation=30, ha="right"); plt.ylabel("Sobol index"); plt.title(f"Sobol (core={core}, N={N_base})"); plt.legend()
    plt.tight_layout(); plt.savefig(fig_path, dpi=160); plt.close()
    return {"plot":str(fig_path), "core":core}

# ---------------- CLI ----------------
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--excel","-e", required=True, type=str, help="Excel workbook")
    ap.add_argument("--mode","-m", required=True, choices=["full","sens","sobol"], help="Run mode")
    ap.add_argument("--core","-c", required=False, choices=["m2","m3"], default="m3", help="Solver core (m2 = one-term clamped-fast; m3 = two-term + root spring)")
    ap.add_argument("--grid", required=False, type=str, default="100x50", help="Contour grid resolution, e.g., 120x60")
    args=ap.parse_args()
    if args.mode=="full":
        out=run_full(args.excel, core=args.core, grid=args.grid); print(json.dumps(out, indent=2))
    elif args.mode=="sens":
        out=run_sens(args.excel, core=args.core); print(json.dumps(out, indent=2))
    else:
        out=run_sobol(args.excel, core=args.core); print(json.dumps(out, indent=2))

if __name__=="__main__":
    main()
