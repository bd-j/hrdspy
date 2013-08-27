import matplotlib.pyplot as pl
import numpy as np
     
#goodfit = (fitter.rp['data_header']['flag'] == 10) & (fitter.data_mag[0,:,2] +18.5 < 20.0)
#glabel = 'V<20 & flag=10'

def plot_sptypes(fitter, PAR = 'LOGT', outfile = 'sptype_logt.dat'):
    #    stypes = np.unique(fitter.rp['data_header']['spType'])

    out = open(outfile,'wb')
    out.write('class I is for class I, II, and III\n')
    out.write('spType  median(T)  sigma(logT)  N_star \n')
    st = fitter.rp['data_header']['spType']
    
    sletters =  ['O','B','A','F','WC','WN']
    snum = ['0','1','2','3','4','5','6','7']
    sclass = ['I','V','W','W']
    for ic in sclass:
        for letter in sletters:
            for num in snum:
                this = ( (np.char.find(st,letter+num) == 0) &
                         np.isfinite(fitter.parval[PAR][0,:,1]) &
                         (fitter.max_lnprob[0,:]*(-2) < 400) &
                         (np.char.find(st,ic) >= 0)
                    )
                if this.sum() == 0 : continue
                out.write('{0:6s}  {1:5.0f}  {2:4.3f}  {3:2d}\n'.format( letter+num+ic,
                                                    10**np.median(fitter.parval[PAR][0,this,1]),
                                                    np.std(fitter.parval[PAR][0,this,1]),
                                                    this.sum() )
                    )

    out.close()

def plot_pars(fitter, PAR1 = 'LOGT', PAR2 = 'LOGL', goodfit = None, glabel = 'good', outfigname = None, loc = 1):
    pl.figure()
    pl.plot(fitter.stargrid.pars[PAR1],
            fitter.stargrid.pars[PAR2],
            'bo',alpha = 0.05, label = 'Prior', mec = 'b'
        )
    pl.plot(fitter.parval[PAR1][0,:,1],
            fitter.parval[PAR2][0,:,1],
            'ro',alpha = 0.2, label = 'all stars', mec = 'r'
        )
    if goodfit is not None:
        pl.plot(fitter.parval[PAR1][0,goodfit,1],
                fitter.parval[PAR2][0,goodfit,1],
                'go',alpha = 0.5, label = glabel, mec = 'g'
            )
    pl.xlabel(PAR1)
    pl.ylabel(PAR2)
    pl.legend(loc = loc)
    if outfigname is None:
        pl.show()
    else:
        pl.savefig(outfigname+'.png')

def plot_precision(fitter, PAR = 'LOGT', versus = None, vlabel = '', goodfit = None, glabel = 'good', outfigname = None):
    if versus is None :
        versus = fitter.parval[PAR][0,:,1]
        vlabel = PAR

    pl.figure()
    pl.plot(versus,
            fitter.parval[PAR][0,:,2]-fitter.parval[PAR][0,:,1],
            'ro',alpha =0.1, label = 'all stars', mec = 'r'
            )
    pl.plot(versus,
            fitter.parval[PAR][0,:,0]-fitter.parval[PAR][0,:,1],
            'ro',alpha =0.1, mec = 'r'
            )
    if goodfit is not None:
        pl.plot(versus[goodfit],
                fitter.parval[PAR][0,goodfit,2]-fitter.parval[PAR][0,goodfit,1],
                'go',alpha =0.5, label = glabel, mec = 'g'
            )
        pl.plot(versus[goodfit],
                fitter.parval[PAR][0,goodfit,0]-fitter.parval[PAR][0,goodfit,1],
                'go',alpha =0.5,  mec = 'g'
            )
        
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

    pl.figure()
    for i, iband in enumerate(bands):
        bname = fitter.rp['fnamelist'][iband].replace('bessell_','')
        pl.plot(fitter.delta_best[0,goodfit,iband],
                versus,
                'o',color = colors[i], alpha = (1.0/len(bands)), mec = colors[i],
                label = bname
            )
        pl.yscale('log')
        pl.ylabel(vlabel)
        pl.xlabel( r'$\Delta {0}$'.format(bname) )

    pl.legend()
    if outfigname is None:
        pl.xlabel(r'$m_{best}-m_{obs}$')
        pl.show()
    else:
        pl.savefig(outfigname+'.png')





