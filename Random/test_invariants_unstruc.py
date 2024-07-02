# Test Invariants and Geometric Terms for Unstructured Schemes
print("Importing modules. May take a while since compiling Julia code.")
import unstruc_mesh_functions as ufn
from get_hicken_op import getOps2D, getOps3D
import numpy as np

# the following is temporary code so that I can make changes to ufn module
import importlib
importlib.reload(ufn)

#TODO: kopriva is p+d-1 in 2d (defaults to direct, essentially)
#TODO: kopriva often sees p+d-1 in 3d, frustratingly, even though it also sometimes doesnt.

# global parameters
dim = 3
p = 4
op_type='Omega' # 'Omega', 'DiagE', 'Gamma' #TODO: add option for DG and DG_rand where metrics are computed using these operators before interpolated to SBP nodes
fac_op_type='lg' # for 2d, 'lg' or 'lgl'. for 3d, any of the above.
vol_quad=2*p # either 2p or potentially 2p-1 if diagE

jac_method = 'exact' # 'deng', 'deng_exa', 'backout', 'direct', exact
met_method = 'chan_ThomasLombard' # 'ThomasLombard', 'Direct', 'exact', 'kopriva_thomaslombard', 'direct_kopriva', 'chan', 'optimized'
metf_method = 'extrapolate' # 'extrapolate', 'exact' 
optimize_met = False
use_pinv = False # if True, uses pseudoinverse of Vandermonde for projection. Otherwise, use quadrature-based projection.
chan_op_type='Modal' # 'Modal', 'Modal_Rand', 'DiagE', 'Omega', 'Gamma'
chan_fac_op_type='lg'
chan_skipp = False # whether to jump straight from xi to p+1 rather than interpolate to p then p+1
chan_init_x = 'x' # start from 'x' (sbp), 'xp1' (p+1 operator nodes), 'interp_p', or 'interp_p1'

warp_factor = 0.15
warp_type = 'ddrf'
xmin = np.array([0.,0.,0.])
xmax = np.array([1.,1.,1.])
plot = True
print_values = True
save_value = True
printtest = True
plotmesh = False


dom_len = xmax - xmin
mesh_dir = 'meshes/'
if dim==2:
    Dx, Dy, Qx, Qy, H, Ex, Ey, xi, xif, B, Nxi, R = getOps2D(p, vol_quad, op_type, fac_op_type)
    Dxi, Qxi, Exi = np.zeros((2,*Dx.shape)), np.zeros((2,*Qx.shape)), np.zeros((2,*Ex.shape))
    Dxi[0,:,:], Dxi[1,:,:] = Dx, Dy
    Qxi[0,:,:], Qxi[1,:,:] = Qx, Qy
    Exi[0,:,:], Exi[1,:,:] = Ex, Ey
    nf = 3
    dom_vol = dom_len[0] * dom_len[1]
elif dim==3:
    Dx, Dy, Dz, Qx, Qy, Qz, H, Ex, Ey, Ez, xi, xif, B, Nxi, R = getOps3D(p, vol_quad, op_type, fac_op_type)
    Dxi, Qxi, Exi = np.zeros((3,*Dx.shape)), np.zeros((3,*Qx.shape)), np.zeros((3,*Ex.shape))
    Dxi[0,:,:], Dxi[1,:,:], Dxi[2,:,:] = Dx, Dy, Dz
    Qxi[0,:,:], Qxi[1,:,:], Qxi[2,:,:] = Qx, Qy, Qz
    Exi[0,:,:], Exi[1,:,:], Exi[2,:,:] = Ex, Ey, Ez
    nf = 4
    dom_vol = dom_len[0] * dom_len[1] * dom_len[2]
else:
    raise ValueError("invalid dim. try 2 or 3.")

"""
if p==2:
    if dim == 2:
        grids = ['square_lev01.msh','square_lev02.msh','square_lev03.msh','square_lev04.msh','square_lev05.msh','square_lev06.msh']
    else:
        grids = ['cube_lev01.msh','cube_lev02.msh','cube_lev03.msh','cube_lev04.msh','cube_lev05.msh']
if p==3:
    if dim == 2:
        grids = ['square_lev01.msh','square_lev02.msh','square_lev03.msh','square_lev04.msh','square_lev05.msh','square_lev06.msh']
    else:
        grids = ['cube_lev01.msh','cube_lev02.msh','cube_lev03.msh','cube_lev04.msh','cube_lev05.msh']
if p==4:
    if dim == 2:
        grids = ['square_lev01.msh','square_lev02.msh','square_lev03.msh','square_lev04.msh','square_lev05.msh','square_lev06.msh']
    else:
        grids = ['cube_lev01.msh','cube_lev02.msh','cube_lev03.msh','cube_lev04.msh','cube_lev05.msh']
"""
if dim == 2:
    grids = ['square_lev01.msh','square_lev02.msh','square_lev03.msh','square_lev04.msh','square_lev05.msh']
else:
    grids = ['cube_lev01.msh','cube_lev02.msh','cube_lev03.msh','cube_lev04.msh','cube_lev05.msh']

if printtest:
    print('Test SBP operators')
    if dim==2:
        ufn.test_operator_2d(Dxi[0],Dxi[1],xi[0],xi[1])
        ufn.test_Exi_2d(xi[0],Exi)
        ufn.test_quad_2d(H,xi[0],facet=False)
        ufn.test_quad_1d(B,xif[0,:,0])
        ufn.test_extrap_2d(R[:,:,0],xi[0],xi[1],xif[0,:,0],xif[1,:,0])
    else:
        ufn.test_operator_3d(Dxi[0],Dxi[1],Dxi[2],xi[0],xi[1],xi[2])
        ufn.test_Exi_3d(xi[0],Exi)
        ufn.test_quad_3d(H,xi[0])
        ufn.test_quad_2d(B,xif[0,:,0],facet=True)
        ufn.test_extrap_3d(R[:,:,0],xi[0],xi[1],xi[2],xif[0,:,0],xif[1,:,0],xif[2,:,0])


transformation = lambda xa: ufn.calc_x(xa,warp_factor,warp_type,dim,xmin,xmax)
transformation_der = lambda xa, dxadxi: ufn.calc_dxdxi(xa,dxadxi,warp_factor,warp_type,dim,xmin,xmax)

###############################################################################
###############################################################################

levstart = 0
hs = np.zeros(len(grids))
hs_linf = np.zeros(len(grids))
hs_apprx = np.zeros(len(grids))
met_ers = np.zeros(len(grids))
met_ers_linf = np.zeros(len(grids))
surf_met_ers = np.zeros(len(grids))
surf_met_ers_linf = np.zeros(len(grids))
jac_ers = np.zeros(len(grids))
jac_ers_linf = np.zeros(len(grids))
norm_ers = np.zeros(len(grids))
norm_ers_linf = np.zeros(len(grids))
phys_op_ers = np.zeros(len(grids))
phys_op_ers_linf = np.zeros(len(grids))
vol_inv = np.zeros(len(grids))
vol_inv_linf = np.zeros(len(grids))
surf_inv = np.zeros(len(grids))
surf_inv_linf = np.zeros(len(grids))

# general structure:
# x: (dim, node, element)
# xfaffine: (dim, node, facet, element)
# dxdxi: (x dim, xi dim, node, element)
# dxdxif: (x dim, xi dim, node, facet, element)
# metrics: (xi dim, x dim, node, element)

for levi in range(len(grids)):
    print("-----------------------------------------------------------------")

    # load the mesh
    gridname = mesh_dir + grids[levi]
    elem_vertices_tags, xaff, xfaff, aff_map = ufn.load_mesh(gridname,xi,xif,dim)
    dxaffdxi = aff_map[:,:dim]
    n_elem = int(len(elem_vertices_tags)/(dim+1))
    
    # get the new physical coordinates for each mesh & facet + the exact inverse metrics
    print("   ... transforming the ..." )
    x = transformation(xaff)
    xf = transformation(xfaff)
    dxdxi_ex = transformation_der(xaff,dxaffdxi)
    dxdxif_ex = transformation_der(xfaff,dxaffdxi)

    if printtest:
        er = np.einsum('lij,mje->mlie', Dxi, x, optimize='optimal') - dxdxi_ex
        print(f"Test: error in approx. inverse metrics: {np.max(np.abs(er)):.1e}")

    if plotmesh:
        ufn.plot_mesh(gridname,x,xf,dim,transformation)

    # compute the exact metrics, jacobians, and normals
    print("   ... computing the exact geometric terms ..." )
    metrics_exa, jac_exa = ufn.calc_met_exa(dxdxi_ex,dim)
    Hphys_exa = np.einsum('ne,n->ne',jac_exa,H)
    metricsf_exa, jacf_exa = ufn.calc_met_exa(dxdxif_ex,dim)
    normals_exa, normal_facs_exa = ufn.calc_normal_exa(metricsf_exa,Nxi,dim)
    if np.any(jac_exa <= 0) or np.any(np.isnan(jac_exa)):
        raise Exception('Transformation yielded an invalid Jacobian. Try a different warp_factor.')
    
    # compute the approximate metrics using the appropriate approach
    print("   ... computing the approximate geometric terms ..." )
    if met_method.lower() == 'exact':
        metrics = metrics_exa
    elif met_method.lower() == 'thomaslombard':
        if dim==2:
            print('WARNING: for dim=2, the Thomas and Lombard approach is not defined. Defaulting to Direct.')
            metrics = ufn.calc_direct(Dxi,x,dim)
        else:
            metrics = ufn.calc_thomaslombard(Dxi,x)
    elif met_method.lower() == 'vinokuryee':
        if dim==2:
            print('WARNING: for dim=2, the Vinokur and Yee approach is the same as Direct.')
            metrics = ufn.calc_direct(Dxi,x,dim)
        else:
            metrics = ufn.calc_vinokuryee(Dxi,x)
    elif met_method.lower() == 'kopriva_thomaslombard':
        if dim==2:
            print('WARNING: for dim=2, the Thomas and Lombard variant is not defined. Defaulting to symmetric form.')
        metrics = ufn.calc_kopriva(Dxi,H,x,xi,p,False,dim,use_pinv)
    elif met_method.lower() == 'kopriva' or met_method.lower() == 'kopriva_vinokuryee':
        metrics = ufn.calc_kopriva(Dxi,H,x,xi,p,True,dim,use_pinv)
    elif met_method.lower() == 'chan_thomaslombard':
        if dim==2:
            print('WARNING: for dim=2, the Thomas and Lombard variant is not defined. Defaulting to symmetric form.')
        metrics = ufn.calc_chan(Dxi,H,x,xi,p,False,dim,chan_op_type,chan_fac_op_type,
                                use_pinv,chan_skipp,chan_init_x,transformation,aff_map)
    elif met_method.lower() == 'chan' or met_method.lower() == 'chan_vinokuryee':
        metrics = ufn.calc_chan(Dxi,H,x,xi,p,True,dim,chan_op_type,chan_fac_op_type,
                                use_pinv,chan_skipp,chan_init_x,transformation,aff_map)
    elif met_method.lower() == 'direct':
        metrics = ufn.calc_direct(Dxi,x,dim)
    elif met_method.lower() == 'direct_project':
        metrics = ufn.calc_direct_project(Dxi,H,x,xi,p,dim,use_pinv)
    else:
        options = ['exact','thomaslombard','vinokuryee','kopriva_thomaslombard','kopriva_vinokuryee','chan_thomaslombard','chan_vinokuryee','direct','direct_kopriva']
        raise ValueError(f"Invalid met_method option. Try one of: {', '.join(options)}")
    
    if metf_method.lower() == 'exact':
        metricsf = metricsf_exa
    elif metf_method.lower() == 'extrapolate':
        metricsf = np.einsum('ijf,lmje->lmife',R,metrics,optimize='optimal')
    else:
        options = ['exact','extrapolate']
        raise ValueError(f"Invalid metf_method option. Try one of: {', '.join(options)}")


    #if np.any(jac < 0) or np.any(np.isnan(jac)):
    #    print('WARNING: There are negative jacobians at {0} nodes!'.format(len(np.argwhere(jac_exa<0))))
    #    levstart = levi+1

    # now compute errors and save
    print("   ... computing the errors ..." )
    hs[levi] = np.mean(jacf_exa/normal_facs_exa)
    hs_linf[levi] = np.max(jacf_exa/normal_facs_exa)
    hs_apprx[levi] = (dom_vol / n_elem)**(1/dim)
    print(f"Approximating element size with h_avg={hs[levi]:.1e}, h_inf={hs_linf[levi]:.1e}")
    
    # save all the errors
    met_ers[levi] = np.max(np.sqrt(np.einsum('ie,lmie->lm', Hphys_exa, (metrics - metrics_exa)**2)))
    met_ers_linf[levi] = np.max(abs(metrics - metrics_exa))
    vol_inv_raw, vol_term, extrap_term = ufn.calc_vol_invariants(Dxi,H,R,B,Nxi,metrics,metricsf)
    vol_inv[levi] = np.max(np.sqrt(np.einsum('ie,mie->m', Hphys_exa, vol_inv_raw**2)))
    vol_inv_linf[levi] = np.max(abs(vol_inv_raw))

    if print_values:
        print(f"Volume metrics error: H-norm={met_ers[levi]:.1e}, L^inf-norm={met_ers_linf[levi]:.1e}")
        print(f"Volume invariants: = H-norm={vol_inv[levi]:.1e}, L^inf-norm={vol_inv_linf[levi]:.1e}")
        print(f"       (volume term L^inf={np.max(abs(vol_term)):.1e}, extrapolation term L^inf={np.max(abs(extrap_term)):.1e})")

print("=================================================================")
if np.all(met_ers<1e-11):
        print('--- !!!! ---')
        print('!!! NOTE !!!: Not calculating volume metrics convergence because EXACT.')
        print('--- !!!! ---')
else:
    _, met_conv = ufn.calc_convergence(hs,met_ers,print_values,'Volume Metric Error H-norm','avg(h)')
    _, met_conv_linf = ufn.calc_convergence(hs_linf,met_ers_linf,print_values,'Volume Metric Error L_inf-norm','max(h)')
    _, met_conv_apprx = ufn.calc_convergence(hs_apprx,met_ers,print_values,'Volume Metric Error H-norm','approx h')

if np.all(vol_inv<1e-11):
        print('--- !!!! ---')
        print('!!! NOTE !!!: Not calculating volume invariants convergence because EXACT.')
        print('--- !!!! ---')
else:
    _, vol_inv_conv = ufn.calc_convergence(hs,vol_inv,print_values,'Volume Metric Error H-norm','avg(h)')
    _, vol_inv_conv_linf = ufn.calc_convergence(hs_linf,vol_inv_linf,print_values,'Volume Metric Error H-norm','max(h)')

if plot:
    # Define the strings using raw strings (r'...')
    h_str = r'$h = \overline{\left( \frac{\mathcal{J}}{\mathcal{J}^f} \right)}$'
    h_linf_str = r'$h = \max \left( \frac{\mathcal{J}}{\mathcal{J}^f} \right)$'
    h_apprx_str = fr'$h = \frac{{1}}{{N_\mathrm{{elem}}^{{1/{dim}}}}}$'
    Hnorm_str = r'$\left\| \mathrm{er} \right\|_{\mathcal{J} H}$'
    linf_str = r'$\left\| \mathrm{er} \right\|_{L^\infty}$'
    if np.all(met_ers<1e-11):
            print('--- !!!! ---')
            print('!!! NOTE !!!: Not plotting volume metrics because EXACT.')
            print('--- !!!! ---')
    else:
        ufn.plot_conv(hs, met_ers, r'Volume Metric Error $H$-norm', h_str, Hnorm_str)
        ufn.plot_conv(hs_linf, met_ers_linf, r'Volume Metric Error $L^\infty$-norm', h_linf_str, linf_str)
        ufn.plot_conv(hs_apprx, met_ers, r'Volume Metric Error $H$-norm', h_apprx_str, Hnorm_str)
    if np.all(vol_inv<1e-11):
        print('--- !!!! ---')
        print('!!! NOTE !!!: Not plotting volume invariants because EXACT.')
        print('--- !!!! ---')
    else:
        ufn.plot_conv(hs_apprx, vol_inv, r'Volume Invariants $H$-norm', h_str, Hnorm_str)
        ufn.plot_conv(hs_apprx, vol_inv_linf, r'Volume Invariants $L^\infty$-norm', h_linf_str, linf_str)

