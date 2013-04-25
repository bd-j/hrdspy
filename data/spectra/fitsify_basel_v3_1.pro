;goto,arrays
fn='wlbc99_'+['p05']+'.cor'
close,1
openr,1,fn

ii=0d

mm=djs_readlines(fn)
nl=n_elements(mm)

data=fltarr(nl*7)
for i=0d,nl-1 do begin
   k=float(strsplit(mm[i],/extract,count=count))
;   print,count
   if count GT 0 then data[ii:ii+count-1]=k
   ii=ii+count
endfor
close,1

nstar = (ii-1221)/1227.

ca=!lightspeed

wave=data[0:1220]
nw=n_elements(wave)
arrays:
;stop

;nstar=458 ;453 for BaSeL 2.2
s={id:0,logt:0,logg:0,logl:0.,m_h:0,v_turb:0,xh:0,wave:0,f_lambda:fltarr(nw),wavelength:fltarr(nw)}
s=replicate(s,nstar)

flux=fltarr(nw,nstar)
stardat=fltarr(6,nstar)
for f=0d,nstar-1 do begin
   s[f].wavelength = wave*10 ;angstroms
   s[f].f_lambda = data[(f+1)*1227:(f+1)*1227+nw-1]*(ca/(s[f].wavelength)^2)  ;mW/m^2/Hz=erg/s/Hz
   stardat=data[(f+1)*1227-6:(f+1)*1227-1]
   s[f].id=stardat[0]
   s[f].logt = alog10(stardat[1])
   s[f].logg = stardat[2]
   s[f].m_h = stardat[3]
   s[f].v_turb = stardat[4]
   s[f].xh = stardat[5]
endfor

mwrfits,s,repstr(fn,'.','_')+'.fits',/create

end
