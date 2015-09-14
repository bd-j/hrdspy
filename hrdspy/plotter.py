import matplotlib.pyplot as pl
import numpy as np
     
#goodfit = (fitter.rp['data_header']['flag'] == 10) & (fitter.data_mag[0,:,2] +18.5 < 20.0)
#glabel = 'V<20 & flag=10'


def catalog_to_fitter(catalog):
    """Take a saved starprops catalog and convert it into
    a barebones fitter object for plotting purposes."""
    return fitter

def rectify_sptypes(st, letters = {'O':10,'B':20.,'A':30.,'F':40.,'G':50.,'WC':100,'WN':200.}):
    """Convert spectral types into numerical values for plotting.
    Includes flagging for uncertain types"""
    
    spnums = np.zeros(len(st))-99
    spclass = np.zeros(len(st))
    spflag = np.zeros(len(st))
    nums = ['','0','1','2','3','4','5','6','7','8','9']
    half = ['','.5']
    classes = ['I','II','III','IV','V']

    for i, c in enumerate(classes):
        this = (np.char.find(st,c) > 0)
        spclass[this] = i
    for letter in letters:
        for num in nums:
            for h in half:
                if num+h is '': continue
                this = (np.char.find(st,letter+num+h) == 0)
                #print(num+h)
                spnums[this] = letters[letter] + float(num+h)

    this = (np.char.find(st,':') >=0)
    spflag[this] = 1
    this = (np.char.find(st,'-') >=0)
    spflag[this] = 2
    return spnums, spclass ,spflag

def plot_sptypes(fitter, PAR = 'LOGT', outfile = 't_vs_type.png', cpar = None):
    
    letters = {'O':10,'B':20.,'A':30.,'F':40.,'G':50.,'WC':100,'WN':200.}
    spnum, spclass, spflag = rectify_sptypes(fitter.rp['data_header']['spType'], letters = letters)
    errors = [fitter.parval[PAR][0,:,1]-fitter.parval[PAR][0,:,0],
              fitter.parval[PAR][0,:,2]-fitter.parval[PAR][0,:,1]]
    rr = np.random.uniform(-0.5,0.5,len(spnum))

    if cpar is None:
        cpar = spclass+1
    elif type(cpar) is str:
        cpar = fitter.parval[cpar][0,:,1]
    else:
        cpar = cpar

    pl.figure()
    pl.scatter(spnum+rr,fitter.parval[PAR][0,:,1], marker = 'o',
               #               mec = None,
               c = cpar, zorder = 100,cmap = pl.cm.coolwarm)
    pl.errorbar(spnum+rr,fitter.parval[PAR][0,:,1],yerr = errors,
                mew = 0, marker = None, fmt = None, zorder = 0,
                ecolor = 'k'
                #ecolor = (np.vstack([spclass,spclass])+1)/5
        )
    tscale_giant_lmc = np.array([[13, 16, 20, 23, 25, 30, 40],
                                 [43e3, 37e3, 29.75e3, 14.5e3, 13.0e3, 10e3, 7.75e3]])
    tscale_dwarf_lmc = np.array([[13, 16, 20, 23, 25, 30, 40],
                                 [48e3,39e3, 30e3, 18.5e3, 15.2e3, 9.5e3, 7.5e3]])

    pl.colorbar()
    pl.plot(tscale_giant_lmc[0,:], np.log10(tscale_giant_lmc[1,:]), color = 'b')
    pl.plot(tscale_dwarf_lmc[0,:], np.log10(tscale_dwarf_lmc[1,:]), color = 'r')

    xtl = ['{0}0'.format(k) for k in letters.keys()] + ['{0}5'.format(k) for k in letters.keys()]
    xtv = letters.values()+ [v+5 for v in letters.values()]
    pl.xticks(xtv, xtl)
    pl.xlim(9,50)
    pl.ylabel(PAR)
    pl.show()

def plot_one_star_confidence(fitter, PAR1 = 'LOGT', PAR2 = 'LOGL', lnprob_grid = None):
    """Plot 'confidence' intervals on parameters for a given star,
    overlaid on the prior distribution."""
    pl.plot(fitter.stargrid.pars[PAR1],
            fitter.stargrid.pars[PAR2],
            'o', c =lnprob_grid, cmap = pl.cm.coolwarm,alpha = 0.05, mec = None)


def plot_pars(fitter, PAR1 = 'LOGT', PAR2 = 'LOGL', goodfit = None, glabel = 'good', outfigname = None, loc = 1):
    """Plot derived stellar parameters against each other,
    including prior grid and an optional filter. """
    pl.figure()
    pl.plot(fitter.stargrid.pars[PAR1],
            fitter.stargrid.pars[PAR2],
            'bo',alpha = 0.05, label = 'Prior', mec = 'b')
    pl.plot(fitter.parval[PAR1][0,:,1],
            fitter.parval[PAR2][0,:,1],
            'ro',alpha = 0.2, label = 'all stars', mec = 'r')
    if goodfit is not None:
        pl.plot(fitter.parval[PAR1][0,goodfit,1],
                fitter.parval[PAR2][0,goodfit,1],
                'go',alpha = 0.5, label = glabel, mec = 'g')
    pl.xlabel(PAR1)
    pl.ylabel(PAR2)
    pl.legend(loc = loc)
    if outfigname is None:
        pl.show()
    else:
        pl.savefig(outfigname+'.png')

def plot_precision(fitter, PAR = 'LOGT', versus = None, vlabel = '', goodfit = None, glabel = 'good', outfigname = None):
    """Plot upper and lower uncrtainties on derived parameters against the parameter."""
    if versus is None :
        versus = fitter.parval[PAR][0,:,1]
        vlabel = PAR

    pl.figure()
    pl.plot(versus,
            fitter.parval[PAR][0,:,2]-fitter.parval[PAR][0,:,1],
            'ro',alpha =0.1, label = 'all stars', mec = 'r')
    pl.plot(versus,
            fitter.parval[PAR][0,:,0]-fitter.parval[PAR][0,:,1],
            'ro',alpha =0.1, mec = 'r')
    if goodfit is not None:
        pl.plot(versus[goodfit],
                fitter.parval[PAR][0,goodfit,2]-fitter.parval[PAR][0,goodfit,1],
                'go',alpha =0.5, label = glabel, mec = 'g')
        pl.plot(versus[goodfit],
                fitter.parval[PAR][0,goodfit,0]-fitter.parval[PAR][0,goodfit,1],
                'go',alpha =0.5,  mec = 'g')
        
    pl.axhline(linewidth = 2, color = 'k')

    pl.xlabel(vlabel)
    pl.ylabel(r'$\Delta {0}$'.format(PAR))
    pl.legend()
    if outfigname is None:
        pl.show()
    else:
        pl.savefig(outfigname+'.png')


def residuals(fitter, bands = [0], versus = None, vlabel = '', colors = ['r'], outfigname = None, goodfit = None, glabel = 'good',):
    if versus is None :
        versus = fitter.max_lnprob[0,goodfit]*(-2)/4
        vlabel = r'$\chi^{2}/4$'

    
    for i, iband in enumerate(bands):
        pl.figure()
        bname = fitter.rp['fnamelist'][iband].replace('bessell_','')
        pl.plot(fitter.delta_best[0,goodfit,iband],
                versus,
                'o',color = colors[i], alpha = 0.01, mec = colors[i],
                label = bname)
        pl.yscale('log')
        pl.ylabel(vlabel)
        pl.xlabel( r'$\Delta {0}$'.format(bname) )

        pl.legend()
        #if outfigname is None: (1.0/len(bands))
        pl.xlabel(r'$m_{best}-m_{obs}$')
        pl.show()
        #else:
        #pl.savefig(outfigname+'.png')


