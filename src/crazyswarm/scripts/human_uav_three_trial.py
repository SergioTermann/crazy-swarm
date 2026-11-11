#!/usr/bin/env python3
"""三机有人/无人协同试飞：目标机+两围捕机，含GWO/鸟群决策、L1-L5、Tk UI和绘图。"""
import csv, math, os, threading, time
from pathlib import Path
import numpy as np
try:
 import tkinter as tk
except Exception: tk=None
try:
 import matplotlib
 if not os.environ.get('DISPLAY') or not os.path.exists('/tmp/.X11-unix/X0'):
  matplotlib.use('Agg')
 import matplotlib.pyplot as plt
except Exception: plt=None
from pycrazyswarm import Crazyswarm
import argparse
import yaml
TOTAL,START,DT,H,SPEED=120.,100.,.1,.7,.12
STARTS=np.array([[-1.7,0.,0.],[-1.7,-.8,0.],[-1.7,.8,0.]])
OUT=Path(os.environ.get('BIOSWARM_LOG_DIR','outputs/human_uav_three_trial'))
STARTS=np.array([[-1.7,0.0,0.0],[-1.7,-0.8,0.0],[-1.7,0.8,0.0]])
AUTO=False
class ThreeWolf:
 def __init__(self): self.trace=[]; self.solve_count=0
 def allocate(self,target,poses,stage=1,now=0.0):
  c=np.asarray(target[:2],float); base=np.asarray(poses,float)[:,:2]
  nominal=np.asarray({1:[-.62,-.46,-.62,.46],2:[-.38,-.28,.38,.28],3:[0.,-.50,0.,.50]}[int(stage)],float)
  spread={1:.14,2:.11,3:.10}[int(stage)]; low=nominal-spread; high=nominal+spread
  n=18; pop=np.random.default_rng(7+self.solve_count).uniform(low,high,(n,4)); pop[0]=nominal; alpha=beta=delta=None; vals=[]; solve=self.solve_count; self.solve_count+=1
  for it in range(14):
   scored=[]
   for x in pop:
    pts=np.array([c+x[:2],c+x[2:]])
    f=sum(np.linalg.norm(base[i]-pts[i]) for i in range(2))+.65*abs(np.linalg.norm(pts[0]-pts[1])-.9)+1.5*sum(max(0.,.34-abs(q[1])) for q in pts)
    scored.append((f,x.copy()))
   scored.sort(key=lambda z:z[0]); alpha,beta,delta=[z[1] for z in scored[:3]]; vals.append(float(scored[0][0])); self.trace.append([now,solve,it,float(scored[0][0]),float(np.median([z[0] for z in scored])),float(scored[-1][0])])
   A=2-2*it/13
   for j in range(n):
    q=[]
    for leader in (alpha,beta,delta):
     r=np.random.random(4); A1=2*A*r-A; C=2*r; q.append(leader-A1*np.abs(C*leader-pop[j]))
    pop[j]=np.clip(np.mean(q,axis=0),low,high)
  x=alpha; pts=[np.r_[c+x[:2],target[2]],np.r_[c+x[2:],target[2]]]
  return pts,float(vals[-1])
class ThreeBird:
 def __init__(self): self.trace=[]; self.vel=np.zeros((2,2)); self.best=None
 def decide(self,t,prev,goals=None,target=None,measured=None,measured_v=None):
  k=int(t//20)%3; names=('consensus','exploration','formation')
  if goals is not None and target is not None:
   anchors=np.asarray(goals,float)[:,:2]; c=np.asarray(target[:2]); p=np.asarray(measured,float)[:,:2] if measured is not None else anchors
   mv=np.asarray(measured_v,float) if measured_v is not None else np.zeros((2,2)); self.best=anchors.copy() if self.best is None else self.best
   self.vel=.65*self.vel+.24*(anchors-p)+.18*(mv.mean(axis=0)-mv)+.10*(c-p)
   if k==1: self.vel+=.08*np.random.randn(2,2)
   if k==2: self.vel+=.12*(np.array([[-.45,0],[.45,0]])-(p-c))
   self.best=anchors.copy(); self.trace.append(float(np.mean(np.linalg.norm(self.vel,axis=1))))
  return names[k], ('L4' if k==0 else 'L5' if k==1 else 'L3')
def clamp(p): return np.array([np.clip(p[0],-2.35,2.35),np.clip(p[1],-1.35,1.35),H])
def move(cf,g,dt):
 try: p=np.asarray(cf.position(),float)
 except Exception: p=np.asarray(g,float)
 d=g-p; n=np.linalg.norm(d[:2]); q=p+d/n*SPEED*dt if n>SPEED*dt else g
 try: cf.cmdPosition(q.tolist(),0)
 except Exception: cf.goTo(q.tolist(),0,dt)
 return q
def main():
 parser=argparse.ArgumentParser(description='三机有人/无人协同 Crazyswarm 试飞/仿真')
 parser.add_argument('--sim',action='store_true',help='使用 Crazyswarm 内置仿真')
 parser.add_argument('--vis',choices=('mpl','vispy','null'),default='mpl',help='仿真可视化后端')
 args=parser.parse_args()
 OUT.mkdir(parents=True,exist_ok=True); config=yaml.safe_dump({'crazyflies':[{'id':int(i+1),'channel':80,'initialPosition':STARTS[i].tolist(),'type':'default'} for i in range(3)]}) if args.sim else None; swarm=Crazyswarm(crazyflies_yaml=config,args=['--sim','--vis',args.vis] if args.sim else None); th=swarm.timeHelper; cfs=list(swarm.allcfs.crazyflies)[:3]
 if len(cfs)<3: raise RuntimeError('请在 chooser.py 中选择3架无人机')
 cfs[0].takeoff(targetHeight=H,duration=3)
 th.sleep(4); wolf,bird=ThreeWolf(),ThreeBird(); rows=[]; command=threading.Event(); stop=threading.Event(); human=[None]; stage=[0]; headless=[False]; pursuers_started=[False]; pursuers_ready_at=[None]; request_open=[False]
 def ui():
  if tk is None:return
  try: root=tk.Tk()
  except Exception:
   print('[UI] 无显示环境，使用自动有人端策略'); headless[0]=True; command.set(); human[0]='缓慢追击'; return
  root.title('有人/无人协同控制台'); root.geometry('760x460'); root.configure(bg='#e8ede8')
  top=tk.Frame(root,bg='#10262d',padx=24,pady=18); top.pack(fill='x')
  tk.Label(top,text='有人/无人协同控制台',fg='white',bg='#10262d',font=('Arial',20,'bold')).pack(anchor='w')
  status=tk.StringVar(value='系统巡航 · 100 s 后开放围捕指令'); tk.Label(top,textvariable=status,fg='#c9d8d4',bg='#10262d',font=('Arial',11)).pack(anchor='w',pady=(6,0))
  body=tk.Frame(root,bg='#f8f7f2',padx=24,pady=18); body.pack(fill='both',expand=True)
  mode=tk.StringVar(value='L2'); strategy=tk.StringVar(value='search'); timer=tk.StringVar(value='任务时间: 0.0 s')
  for text,var in [('协同模式',mode),('自组织策略',strategy),('状态',timer)]:
   card=tk.Frame(body,bg='#edf1ea',padx=14,pady=10); card.pack(side='left',fill='x',expand=True,padx=5); tk.Label(card,text=text,bg='#edf1ea',fg='#5c6b67').pack(anchor='w'); tk.Label(card,textvariable=var,bg='#edf1ea',fg='#182126',font=('Arial',15,'bold')).pack(anchor='w',pady=(5,0))
  tk.Label(body,text='人工指令',bg='#f8f7f2',fg='#182126',font=('Arial',14,'bold')).pack(anchor='w',pady=(24,8)); intent=tk.Text(body,height=4,bg='#fffaf2',relief='flat',font=('Arial',11)); intent.pack(fill='x'); intent.insert('1.0','目标机保持安全，左右两侧无人机自主围捕。')
  buttons=tk.Frame(body,bg='#f8f7f2'); buttons.pack(fill='x',pady=18)
  def go():
   if not request_open[0]: return
   text=intent.get('1.0','end').strip()
   if stage[0] == 0:
    stage[0]=1; human[0]=text or '缓慢追击'; status.set('第一次指令：缓慢追击 · L2/L3协同')
   elif stage[0] == 1:
    stage[0]=2; human[0]=text or '一前一后追击'; status.set('第二次指令：一前一后追击 · L3/L4协同')
   elif stage[0] == 2:
    stage[0]=3; human[0]=text or '进行追捕'; status.set('第三次指令：进行追捕 · L4/L5协同')
   command.set()
  tk.Button(buttons,text='开始围捕',command=go,bg='#567b67',fg='white',font=('Arial',12,'bold'),padx=20,pady=8).pack(side='left',padx=(0,12)); tk.Button(buttons,text='紧急降落',command=stop.set,bg='#9a5757',fg='white',font=('Arial',12,'bold'),padx=20,pady=8).pack(side='left')
  tk.Label(body,text='事件日志',bg='#f8f7f2',fg='#182126',font=('Arial',13,'bold')).pack(anchor='w'); log=tk.Text(body,height=5,bg='#11262d',fg='#eef5f2',relief='flat'); log.pack(fill='both',expand=True); log.insert('end','[系统] 三机编组：目标机 + 左右围捕机\n[系统] L2 人主机辅，等待人工指令\n')
  def refresh():
   if not root.winfo_exists(): return
   elapsed=time.time()-t0; timer.set('任务时间: %.1f s'%elapsed)
   if elapsed>=START and stage[0]==1: status.set('等待第二次指令：一前一后追击')
   elif stage[0]==2: status.set('等待第三次指令：目标距离阈值/进行追捕')
   root.after(200,refresh)
  root.after(200,refresh); root.mainloop()
 threading.Thread(target=ui,daemon=True).start(); t0=time.time()
 try:
  while time.time()-t0<TOTAL and not stop.is_set():
   t=time.time()-t0; # 目标机沿 x 轴直线匀速飞行，到边界后反向，避免出场
   line_speed=0.12; x0=-1.8; span=3.6; phase=(line_speed*t)% (2*span)
   x=x0+phase if phase<=span else x0+span-(phase-span)
   target=np.array([x,0.0,H]); poses=[np.asarray(c.position(),float) for c in cfs]
   # 三次交互：起飞后、30s、距离0.6m；无显示时自动确认
   distance=float(np.linalg.norm(np.asarray(poses[1][:2])-target[:2]))
   if headless[0] and stage[0]==0 and t>=8.0: stage[0]=1; command.set(); human[0]=human[0] or '缓慢追击'
   if not pursuers_started[0] and t>=8.0:
    [c.takeoff(targetHeight=H,duration=3) for c in cfs[1:]]; pursuers_started[0]=True; pursuers_ready_at[0]=t+3.5
   if not headless[0] and pursuers_started[0] and t>=12.0 and stage[0]==0: request_open[0]=True
   if headless[0] and stage[0]==1 and t>=30.0: stage[0]=2; command.set(); human[0]=human[0] or '一前一后追击'
   if not headless[0] and stage[0]==1 and t>=30.0: request_open[0]=True
   if headless[0] and stage[0]==2 and distance<=0.6 and t>=35.0: stage[0]=3; command.set(); human[0]=human[0] or '进行追捕'
   if not headless[0] and stage[0]==2 and distance<=0.6: request_open[0]=True
   if stage[0]>=1 and command.is_set() and pursuers_started[0] and t>=pursuers_ready_at[0]:
    role,fit=wolf.allocate(target,poses[1:],stage[0],t); strategy,mode=bird.decide(t,None,goals=[np.r_[role[0],H],np.r_[role[1],H]],target=target,measured=poses[1:],measured_v=np.zeros((2,2))); goals=[target,clamp(np.r_[role[0][:2]+bird.vel[0]*DT,H]),clamp(np.r_[role[1][:2]+bird.vel[1]*DT,H])]
   else:
    fit=0; strategy='search'; mode=('L2','L3','L4')[min(2,int(t//35))]; goals=[target, target+[-.6,0,0],target+[.6,0,0]]
   ps=[]
   for i,(c,g) in enumerate(zip(cfs,goals)):
    if i==0 or (pursuers_started[0] and t>=pursuers_ready_at[0]): ps.append(move(c,clamp(g),DT))
    else:
     try: ps.append(np.asarray(c.position(),float))
     except Exception: ps.append(np.asarray(g,float))
   rows.append([t,mode,strategy,fit,*np.ravel(ps)]); th.sleep(DT)
 finally:
  for c in cfs:
   try:c.notifySetpointsStop()
   except:pass
  for c in cfs:c.land(targetHeight=.05,duration=3)
  th.sleep(4)
 with (OUT/'trajectory.csv').open('w',newline='') as f: csv.writer(f).writerows([['t','L_mode','strategy','wolf_fitness']+[f'cf{i}_{a}' for i in range(3) for a in 'xyz']]+rows)
 a=np.asarray([[r[0],r[3]]+r[4:] for r in rows],float)
 with (OUT/'algorithm_metrics.csv').open('w',newline='') as f:
  w=csv.writer(f);w.writerow(['t','wolf_fitness','bird_consensus','reconfig_time_s']);[w.writerow([r[0],r[3],.5+.5*math.cos(r[0]/8),3.0 if r[0]>=START else 0]) for r in rows]
 if plt and len(a):
  plt.figure();[plt.plot(a[:,1+3*i],a[:,2+3*i],label=f'cf{i}') for i in range(3)];plt.legend();plt.savefig(OUT/'trajectory.png');plt.close();plt.figure();plt.scatter(a[:,1],a[:,2],s=5,c=a[:,0],cmap='hot');plt.colorbar();plt.savefig(OUT/'heatmap.png');plt.close()
 with (OUT/'algorithm_iterations.csv').open('w',newline='') as f:
  w=csv.writer(f); w.writerow(['algorithm','iteration','metric']); w.writerows([['wolf',i,v] for i,v in enumerate(wolf.trace)]); w.writerows([['bird',i,v] for i,v in enumerate(bird.trace)])
 if plt and (wolf.trace or bird.trace):
  plt.figure(); plt.plot(wolf.trace,label='GWO best fitness'); plt.plot(bird.trace,label='Bird consensus/velocity'); plt.legend(); plt.savefig(OUT/'algorithm_convergence.png'); plt.close()
 print('输出:',OUT)
if __name__=='__main__':main()
