"""Original periodic implicit-surface animation. Run at repository root with numpy, scipy, Pillow and ffmpeg."""
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from scipy.ndimage import distance_transform_edt,map_coordinates
import subprocess,colorsys
W,H=960,540;N=240;FPS=24
Y,X=np.mgrid[-1.1:1.1:complex(H),-1.96:1.96:complex(W)].astype(np.float32)
def segment(x,y,a,b,r1,r2):
 dx=b[0]-a[0];dy=b[1]-a[1];q=np.clip(((x-a[0])*dx+(y-a[1])*dy)/(dx*dx+dy*dy),0,1)
 return np.sqrt((x-a[0]-q*dx)**2+(y-a[1]-q*dy)**2)-(r1+(r2-r1)*q)
def smoothmin(a,b,k=.13):
 h=np.maximum(k-np.abs(a-b),0)/k
 return np.minimum(a,b)-h*h*k*.25
# Rock: a lopsided mineral volume. Branch: tapering, connected growth.
angle=np.arctan2(Y,X);rad=np.sqrt((X/.97)**2+(Y/1.08)**2)
rock=rad-(.64+.055*np.sin(3*angle)+.025*np.sin(7*angle))
branch=np.full_like(X,10)
for a,b,ra,rb in [((-.28,-.78),(-.05,.48),.13,.10),((-.10,-.12),(-.75,.42),.12,.055),((-.47,.18),(-.82,.76),.075,.035),((-.02,.24),(.57,.8),.10,.04),((.25,.5),(.82,.46),.075,.025),((-.14,-.25),(.59,-.48),.10,.045),((.32,-.39),(.62,-.84),.06,.025)]:branch=smoothmin(branch,segment(X,Y,a,b,ra,rb))
# Bird in flight: two swept wings, a tapered tail, neck, head and beak.
mask=Image.new('L',(W,H));draw=ImageDraw.Draw(mask)
def polygon(points):draw.polygon([((x+1.96)/3.92*(W-1),(y+1.1)/2.2*(H-1)) for x,y in points],fill=255)
polygon([(-.24,.23),(-.73,-.90),(-.48,-.77),(-.31,-.52),(-.10,-.48),(.21,.17)])
polygon([(-.03,.16),(.31,-.64),(.76,-.92),(.61,-.51),(.49,-.43),(.38,-.17),(.27,.24)])
polygon([(-.35,.20),(-.99,.68),(-.72,.24),(-.87,.16),(-.24,.14)])
polygon([(.38,.17),(.60,.04),(.74,.06),(.78,.15),(.96,.21),(.77,.23),(.71,.31),(.54,.29),(.30,.32)])
bmask=np.array(mask)>0
bird=distance_transform_edt(~bmask,sampling=(2.2/H,3.92/W))-distance_transform_edt(bmask,sampling=(2.2/H,3.92/W))
body=(np.sqrt(((X+.03)/.44)**2+((Y-.22)/.15)**2)-1)*.15
bird=smoothmin(bird.astype(np.float32),body,.09)
# A trefoil loop, with depth retained where the ribbon crosses itself.
u=np.linspace(0,2*np.pi,90);kx=.25*(np.sin(u)+2*np.sin(2*u));ky=.25*(np.cos(u)-2*np.cos(2*u));kz=-.15*np.sin(3*u)
knot=np.full_like(X,10);knotheight=np.zeros_like(X)
for n in range(len(u)-1):
 d=segment(X,Y,(kx[n],ky[n]),(kx[n+1],ky[n+1]),.095,.095)
 nearest=d<knot;knotheight=np.where(nearest,(kz[n]+kz[n+1])/2,knotheight);knot=np.minimum(knot,d)
fields=[rock,branch,bird,knot];palette=np.array([colorsys.hsv_to_rgb(h,.82,.86) for h in np.linspace(0,1,2049)],dtype=np.float32)
out=Path('public/art');out.mkdir(exist_ok=True);Path('work').mkdir(exist_ok=True)
pipe=subprocess.Popen(['ffmpeg','-loglevel','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s',f'{W}x{H}','-r',str(FPS),'-i','-','-an','-c:v','libx264','-preset','medium','-crf','19','-pix_fmt','yuv420p','-movflags','+faststart',str(out/'metamorphosis.mp4')],stdin=subprocess.PIPE)
shots=[]
for i in range(N):
 t=2*np.pi*i/N;p=4*i/N;state=int(p)%4;f=p-int(p);blend=f*f*f*(f*(f*6-15)+10)
 rotation=.10*np.sin(t*2);c=np.cos(rotation);s=np.sin(rotation)
 x=X*c-Y*s;y=X*s+Y*c
 wx=x+.027*np.sin(y*5+t)+.012*np.sin(y*11-t*2);wy=y+.025*np.sin(x*5-t)+.01*np.sin(x*13+t*2)
 coords=np.array([(wy+1.1)/2.2*(H-1),(wx+1.96)/3.92*(W-1)])
 a=map_coordinates(fields[state],coords,order=1,mode='nearest');b=map_coordinates(fields[(state+1)%4],coords,order=1,mode='nearest');d=a*(1-blend)+b*blend
 kw=((1-blend) if state==3 else blend if (state+1)%4==3 else 0)
 depth=np.sqrt(np.maximum(-d,0)+.0001)
 grain=np.sin(wx*31+np.sin(wy*13+t))*np.sin(wy*26+np.sin(wx*9-t));folds=np.sin(wx*7+wy*4+t)*np.sin(wy*6-t)
 z=.55*depth+.010*grain+.022*folds
 if kw:z+=kw*.6*map_coordinates(knotheight,coords,order=1,mode='nearest')
 zy,zx=np.gradient(z,2.2/H,3.92/W);norm=np.sqrt(zx*zx+zy*zy+1);nx=-zx/norm;ny=-zy/norm;nz=1/norm
 light=np.array([-.4,-.65,.65]);light/=np.linalg.norm(light);diff=np.clip(nx*light[0]+ny*light[1]+nz*light[2],0,1)
 # Hue travels continuously along the surface and one full turn per loop.
 hue=(.75-i/N+.035*nx+.025*wy+.025*np.sin(wx*4+wy*3-t))%1;idx=np.minimum((hue*2048).astype(int),2047);base=palette[idx]
 shimmer=.5+.5*np.sin(nx*7+ny*5+t*2+folds)
 rgb=base*(.18+.76*diff[...,None])*(.75+.25*shimmer[...,None])
 spec=np.clip(nx*(-.25)+ny*(-.38)+nz*.89,0,1)**42;rgb+=spec[...,None]*(.5+base*.6)
 side=np.clip(nx*.8+ny*.25+nz*.25,0,1)**5
 secondary=palette[((hue+.12)%1*2048).astype(int)];rgb+=side[...,None]*secondary*.6
 rim=np.exp(-(d/.018)**2);rgb+=rim[...,None]*base*.35
 alpha=np.clip(.5-d/.009,0,1);bg=np.zeros((H,W,3),dtype=np.float32)+np.array([.012,.009,.022],np.float32)
 halo=np.exp(-(np.maximum(d,0)/.12)**2)*.018;bg+=halo[...,None]*base
 rgb=np.uint8(np.clip(rgb*alpha[...,None]+bg*(1-alpha[...,None]),0,1)*255);im=Image.fromarray(rgb)
 if i in [0,60,120,180]:shots.append(im.copy())
 if i==0:im.save(out/'metamorphosis-poster.jpg',quality=94)
 pipe.stdin.write(rgb.tobytes())
 if i%60==0:print(f'Rendered state {state+1}/4, frame {i}/{N}',flush=True)
pipe.stdin.close();assert pipe.wait()==0
sheet=Image.new('RGB',(W*2,H*2))
for n,im in enumerate(shots):sheet.paste(im,((n%2)*W,(n//2)*H))
sheet.save('work/metamorphosis-contact.jpg',quality=90)
print('Four-stage 10 second loop ready',flush=True)
