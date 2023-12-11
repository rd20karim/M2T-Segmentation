import numpy as np
import matplotlib.pyplot as plt
import torch
from matplotlib.lines import Line2D
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d.art3d import Line3D
import matplotlib.animation as animation
from matplotlib import rc
import matplotlib
from matplotlib import colors
import os
from matplotlib.transforms import Affine2D


colors = ['sienna','sienna','sienna','sienna','orange','orange','orange','red',"red",'red','blue', 'blue', 'blue', 'blue', 'blue',
          "cyan","cyan","cyan","cyan","cyan"]
kit_graph = [[0,1,2,3,3,5,6,3,8, 9,0,11,12,13,14,0,16,17,18,19],
             [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]]


class SubplotAnimation():
    def __init__(self, poses, frames, use_kps_3d=range(30), down_sample_factor=1,sample = None,pred=None,idxs=None,ref=None,ax=None):
        super().__init__()
        self.paused = False
        self.poses = poses
        self.pred_words = pred
        self.indx_w2p= idxs
        self.w_t_1 = ""
        self.sample = sample
        #self.ax = plt.proj
        #self.ax = plt.axes(projection='3d')
        self.ax = ax
        self.pos_text = self.ax.transData
        self.it = 0
        self.ref = '' if ref is None else ref
        self.s = down_sample_factor  # for down_sampling
        self.t = range(0, int(len(frames) / self.s))
        self.line = Line3D([], [], [], marker='o', color='black', linestyle='', markersize=4)
        self.line_root = Line3D([], [], [], color='white', linestyle='--', markersize=4,linewidth=2)
        self.bones = Line3D([], [], [], color='w', linestyle='-', markersize=3)
        self.pred_len = 0
        self.use_3dkps = use_kps_3d

        if self.sample is not  None:
            self.clip_poses = self.poses[self.sample].reshape(-1, 21, 3)
        else:
            self.clip_poses = self.poses.reshape(-1, 21, 3)

        fc = 2
        min_x,min_z,min_y = fc*self.clip_poses.min(0).min(0)
        max_x,max_z,max_y = fc*self.clip_poses.max(0).max(0)
        print(min_x,min_z,min_y,max_x,max_z,max_y )
        self.ax.set_xlim3d([min_x, max_x])
        self.ax.set_ylim3d([min_y, max_y])
        self.ax.set_zlim3d([min_z, max_z])
        #self.ax.set_adjustable("datalim")

        self.ax.grid = False

        #set the background color to gray
        self.ax.set_facecolor((0.7, 0.7, 0.7))

        # create a plane at the bottom of the human motion foot
        x = np.linspace(min_x, max_x, 10)
        y = np.linspace(min_y, max_y, 10)
        X, Y = np.meshgrid(x, y)
        Z = min_z*np.ones_like(X)#/fc
        self.ax.plot_surface(X, Y, Z, color='k', alpha=0.65)

        self.ax.set_axis_off()
        self.ax.set_xticklabels([])
        self.ax.set_yticklabels([])
        self.ax.set_zticklabels([])
        # Set title descriptions
        self.title = self.ax.set_title('',fontsize=20)
        self.ax.add_line(self.line)
        self.ax.add_line(self.line_root)

        fps = 10

    def draw_frame(self, i,kw,alpha):
        fc = 2 # skeleton scale only for better visualization
        u, p, v = fc * self.clip_poses[self.s * i, 4, :]
        u = u +(-1)**(i%2)*[0,0.1][i%2]
        if self.pred_words is not None and self.indx_w2p is not None:
            pass
            # print(self.indx_w2p, self.pred_words)
            #assert len(self.indx_w2p)==len(self.pred_words)
            # if i in self.indx_w2p:
            #     wi = np.where(np.asarray(self.indx_w2p)==i)[0]
            #     self.w_t_1 = " ".join([self.pred_words[int(k)] for k in wi])
            #     self.pred_len += len(self.w_t_1.split(" "))
            #     print(u,v,p,"frame-->",i)
            #     text = self.ax.text(u, v, p, self.w_t_1,c="black",fontsize = 18, weight='bold')
            #     #self.title.set_text(self.w_t_1)
        assert len(self.indx_w2p) == len(self.pred_words)
        self.title.set_text(self.pred_words[kw])
        x = fc * self.clip_poses[self.s * i, :, 0]
        y = fc * self.clip_poses[self.s * i, :, 1]
        z = fc * self.clip_poses[self.s * i, :, 2]

        xroot = fc * self.clip_poses[:self.s * i, 0, 0]
        yroot = fc * self.clip_poses[:self.s * i, 0, 1]
        zroot = fc * self.clip_poses[:self.s * i, 0, 2]
        min_x, min_z, min_y = fc * self.clip_poses.min(0).min(0)
        # Root trajectory
        self.line_root.set_data_3d(xroot, zroot, yroot)

        #if i in list(range(0,self.t.stop,int(self.t.stop/10))):
        #TODO WORK ON ADD JOINT CONNECTIONS
        starts = kit_graph[0]
        ends = kit_graph[1]
        skeletonFrame = np.vstack([x,y,z]).T
        #self.ax.cla() delete all previous plots

        c = 0
        self.skel_connect = []
        self.bones = []
        for k, j in zip(starts, ends):
            self.skel_connect += self.ax.plot3D([skeletonFrame[k][0], skeletonFrame[j][0]], [skeletonFrame[k][2], skeletonFrame[j][2]],
                    zs=[skeletonFrame[k][1], skeletonFrame[j][1]],c=colors[c],linewidth=3)
            for line in self.skel_connect:line.set_alpha(alpha)
            # self.bones += self.ax.plot3D([skeletonFrame[k][0], skeletonFrame[j][0]], [skeletonFrame[k][2], skeletonFrame[j][2]],
            #         zs=[skeletonFrame[k][1], skeletonFrame[j][1]],c="darkgreen",linewidth=1)
             #for line in self.bones:line.set_alpha(alpha)

            c+=1
        if alpha==1.:
            self.ax.scatter3D(x,z,y,alpha=alpha,c='black',s=10,marker="o")
            for k, j in zip(starts, ends):
                self.skel_connect += self.ax.plot3D([skeletonFrame[k][0], skeletonFrame[j][0]], [skeletonFrame[k][2], skeletonFrame[j][2]],
                        zs=[skeletonFrame[k][1], skeletonFrame[j][1]],c="black",linewidth=2)
        #self.ax.scatter3D(x,z,y,alpha=alpha,c='black',s=alpha,marker="o")
        #self.line.set_data_3d(x, z, y)
        # self.line.set_alpha(alpha)
if __name__=='__main__':
    kitmld = np.load(r"C:\Users\karim\PycharmProjects\SemMotion\src\kitmld_anglejoint_2016_30s_final_cartesian.npz", allow_pickle=True)

    sample = 1556
    normalized_poses = kitmld["normalized_poses"]
    descriptions = [S for S in kitmld["descriptions"]]

    # avoid skipping some frames due to memory limit
    Tmax = 700*4

    for idw in [20,25,28,29,30]:

        fzmotion = SubplotAnimation(normalized_poses, frames=np.arange(0, min(normalized_poses[sample].shape[0], Tmax)),
                                    use_kps_3d=range(21),down_sample_factor=10, sample=sample, pred="A person turns right".split(" "),
                                    idxs=[0,0,idw,idw])

        # Test frozen motion segment visualization

        for i in range(idw-5,idw+5):
            fzmotion.draw_frame(i)
        fzmotion.ax.figure.savefig(f"fz_sample_{sample}_{idw}.png")


