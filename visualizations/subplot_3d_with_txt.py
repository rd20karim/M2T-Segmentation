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

def graph_colors(dataset_name):
    global colors,graph
    if dataset_name=='h3D' :  # HUMAN ML3d GRAPH
        colors = ['sienna','sienna','sienna','sienna','orange','orange','orange','orange',"red",'red', 'red', 'indigo', 'indigo','blue', 'blue',
                  "cyan","cyan","teal","teal","cyan","cyan","cyan","cyan"]
        graph= [[0,2,5,8,0,1,4,7,0,3,6,9,12,9,14,17,19,9,13,16,18],
                     [2,5,8,11,1,4,7,10,3,6,9,12,15,14,17,19,21,13,16,18,20]]
    else:
        colors = ['black','black','black','black','orange','orange','orange','red',"red",'red','blue', 'blue', 'blue', 'blue', 'blue',
                  "cyan","cyan","cyan","cyan","cyan","cyan"]
        graph = [[0,1,2,3,3,5,6,3,8, 9,0,11,12,13,14,0,16,17,18,19],
                     [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]]
class SubplotAnimation(animation.TimedAnimation):
    def __init__(self, poses, frames, use_kps_3d=range(30), down_sample_factor=1,sample = None,pred=None,idxs=None,ref=None,dataset_name='kit2016'):

        graph_colors(dataset_name)

        fig = plt.figure(figsize=(10,8),facecolor=(0.7,0.7,0.7))
        n_joint = len(use_kps_3d)
        #fig.set_size_inches(8,8)
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        #fig.subplots_adjust(wspace=-0.1)
        self.paused = False
        self.poses = poses
        self.pred_words = pred
        self.indx_w2p= idxs
        self.w_t_1 = ""
        self.sample = sample
        self.ax = plt.axes(projection='3d')
        self.pos_text = fig.transFigure #self.ax.figure.transFigure
        self.it = 0
        self.ref = '' if ref is None else ref
        self.s = down_sample_factor  # for down_sampling
        self.t = range(0, int(len(frames) / self.s))
        self.line = Line3D([], [], [], marker='o', color='blue', linestyle='', markersize=3)
        self.line_root = Line3D([], [], [], color='w', linestyle='--', markersize=4,linewidth=3)
        self.bones = Line3D([], [], [], color='w', linestyle='-', markersize=3)
        self.data_name = dataset_name
        self.pred_len = 0
        self.use_3dkps = use_kps_3d


        if self.sample is not  None:
            self.clip_poses = self.poses[:,self.sample,:].reshape(-1, n_joint, 3)
        else:
            self.clip_poses = self.poses.reshape(-1, n_joint, 3)

        if self.data_name=='h3D':
            MINS = self.clip_poses.min(axis=0).min(axis=0)
            MAXS = self.clip_poses.max(axis=0).max(axis=0)
            height_offset = MINS[1]
            self.clip_poses[:, :, 1] -= height_offset

        if "2016" in dataset_name:
            min_y,min_z,min_x = -1,-1,-1
            max_x,max_z,max_y = 1,1,1
        else:
            r = 4  # if self.data_name=='h3D' else 1
            min_y, min_z, min_x = -r, -r, -r
            max_x, max_z, max_y = r, r, r
        self.ax.set_xlim3d([min_x, max_x])
        self.ax.set_zlim3d([min_z, max_z])

        self.coordinate_start_text = (min_x,0,max_z)
        self.ax.grid = False
        plt.axis('off')
        self.ax.set_xticklabels([])
        self.ax.set_yticklabels([])
        self.ax.set_zticklabels([])

        #set the background color to gray
        self.ax.set_facecolor((0.7, 0.7, 0.7))

        # create a plane at the bottom of the human motion feet
        fsc =  1.25
        x = np.linspace(fsc*min_x, fsc*max_x, 10)
        y = np.linspace(fsc*min_y, fsc*max_y, 10)
        if self.data_name=='h3D':
            self.ax.view_init(elev=120, azim=-90,vertical_axis='z')
            self.ax.dist = 10 # control the distance of 3D space from the camera
            z = np.linspace(fsc * min_z, fsc * max_z, 10)
            X, Z = np.meshgrid(x, z)
            min_foot = self.clip_poses[:, :, 1].min()
            Y = fsc * min_foot * np.ones_like(X)
            # light = matplotlib.colors.LightSource(azdeg=-90, altdeg=120)
            # illuminated_surface = light.shade(Y, cmap=matplotlib.cm.twilight)
            self.ax.plot_surface(X, Y, Z, color='k', alpha=0.3)  # ,facecolors=illuminated_surface
        else:
            X, Y = np.meshgrid(x, y)
            min_foot = self.clip_poses[:, :, 1].min()
            Z = fsc*min_z*np.ones_like(X) if '2016' in dataset_name else fsc*min_foot*np.ones_like(X)
            self.ax.plot_surface(X,Y,Z, color='k', alpha=0.3)

        #Set title descriptions
        self.title = self.ax.set_title('',fontsize=18)
        self.ax.add_line(self.line)
        self.ax.add_line(self.line_root)
        self.k=0
        self.ax.set_ylim3d([min_y, max_y])

        fps = 10
        self.animation = animation.TimedAnimation.__init__(self, fig, interval= int(1000 / fps), blit=True)
        fig.canvas.mpl_connect('button_press_event', self.toggle_pause)

    def toggle_pause(self, *args, **kwargs):
        if self.paused:
            self.animation.resume()
        else:
            self.animation.pause()
        self.paused = not self.paused

    def _draw_frame(self, i):
        fc  = 2.5 if  "2016" in self.data_name else 1 # skeleton scale only for better visualization
        #self.title.set_text(' Frame={} '.format(i))
        #self.f = 0
        if i>=1:   self.ax.figure.texts.remove(self.old_text)
        self.old_text =  self.ax.figure.text(0.38,0.95, ' Frame={} '.format(i),fontsize=25,weight='bold')
        canvas = self.ax.figure.canvas
        #v = 0
        if self.pred_words is not None and self.indx_w2p is not None:
            #print(self.indx_w2p, self.pred_words)
            assert len(self.indx_w2p)==len(self.pred_words)
            if i in self.indx_w2p:
                wi = np.where(np.asarray(self.indx_w2p)==i)[0]
                self.w_t_1 = " ".join([self.pred_words[int(k)] for k in wi])
                self.pred_len += len(self.w_t_1.split(" "))
                #np.random.rand(3)
                sep = " "
                if self.pred_len<6 or self.k==0:
                    self.f = 0; self.k = 1
                if self.pred_len >= 6 and self.k==1:
                    self.pos_text = self.ax.figure.transFigure; self.k = -1 ; self.f = 1
                if self.pred_len >= 12 and self.k==-1:
                    self.pos_text = self.ax.figure.transFigure; self.k = 2
                    self.f = 2

                shift= self.f*0.07

                text_colors = ['red', 'black', 'darkorange', 'green', 'blue', 'darkviolet']
                # text = self.ax.text(u,v,p,self.w_t_1+sep,transform=self.pos_text,
                #                     c=text_colors[self.it%len(text_colors)],
                #                     fontsize = 22,weight='bold')
                # print(self.w_t_1+sep)
                print(self.w_t_1+sep)
                text = self.ax.figure.text(x=0,y=0.85-shift,s=self.w_t_1+sep,transform=self.pos_text,
                                            c=text_colors[self.it%len(text_colors)],fontsize = 25,weight='bold')
                text.draw(canvas.get_renderer())
                ex = text.get_window_extent()
                self.pos_text  = text.get_transform() + Affine2D().translate(ex.width, 0)
                self.title.set_text(self.w_t_1)
                self.it += 1

        x = fc * self.clip_poses[self.s * i, :, 0]
        y = fc * self.clip_poses[self.s * i, :, 1]
        z = fc * self.clip_poses[self.s * i, :, 2]


        xroot = fc * self.clip_poses[:self.s * i, 0, 0]
        yroot = fc * self.clip_poses[:self.s * i, 0, 1]
        zroot = fc * self.clip_poses[:self.s * i, 0, 2]


        starts = graph[0]
        ends = graph[1]
        skeletonFrame = np.vstack([x,y,z]).T
        #self.ax.cla() delete all previous plots
        try:
            for line in self.skel_connect: self.ax.lines.remove(line)
            self.skel_connect = []
        except AttributeError:
            self.skel_connect = []
        except ValueError:
            pass
        alphas = np.linspace(0,1,5)
        c = 0

        if self.data_name=='h3D' :
            m,n,p = [0,1,2]
            self.line.set_data_3d(x, y, z)
            self.line_root.set_data_3d(xroot, yroot, zroot)

        else :
            m,n,p= [0,2,1]
            self.line.set_data_3d(x, z, y)
            self.line_root.set_data_3d(xroot, zroot, yroot)

        self.skel_connect = []
        for k, j in zip(starts, ends):
            self.skel_connect += self.ax.plot3D([skeletonFrame[k][m], skeletonFrame[j][m]], [skeletonFrame[k][n], skeletonFrame[j][n]],[skeletonFrame[k][p], skeletonFrame[j][p]],c=colors[c%(len(colors)-1)],linewidth=4.5)
            #for line in self.skel_connect:line.set_alpha(alphas[i%5])
            c+=1

    def new_frame_seq(self):
        return iter(range(len(self.t)))

    def _init_draw(self):
        self.line.set_data_3d([], [], [])

if __name__=='__main__':
    dataset_name ='kit' # 'kit'
    # ------------------------------------ For KIT-ML Original ANIMATIONS -----------------------------------------------------------------
    if dataset_name == 'kit2016':

        kitmld = np.load(r"C:\Users\karim\PycharmProjects\SemMotion\datasets\kitmld_anglejoint_2016_30s_final_cartesian.npz", allow_pickle=True)

        sample = 1563
        normalized_poses = kitmld["kitmld_array"]

        descriptions = [S for S in kitmld["descriptions"]]
        # avoid skipping some frames due to memory limit
        matplotlib.rcParams['animation.embed_limit'] = 2 ** 128

        Tmax = 700*4
        ani = SubplotAnimation(normalized_poses/1000,frames=np.arange(0, min(normalized_poses[sample].shape[0], Tmax)), use_kps_3d=range(21),
                               down_sample_factor=1,sample=sample,pred="A person turns something right left to the ground".split(" "),idxs=[7*i for i in range(9)])

        # """"save animation
        rc('animation', html='jshtml')
        name_file = f"test_sample_{sample}.gif"
        ani.save(name_file)
        print("Animation saved in "+name_file)#os.getcwd()+

    #------------------------------------ For KIT-ML ANIMATIONS -----------------------------------------------------------------
    if dataset_name == 'kit':
        n_joint = 21
        sample = 43 # 1437
        kit_data = np.load("../datasets/kit_with_splits_2023.npz", allow_pickle=True)
        data = np.asarray([xyz.reshape(-1, n_joint * 3) for xyz in kit_data['kitmld_array']], dtype=object)
        descriptions = kit_data['descriptions']
        #data = kitmld['kitmld_array']

    #---------------------------------- For Human-ML3D ANIMATIONS --------------------------------------------------------------
    if dataset_name == 'h3D':
        n_joint = 22
        sample = 43
        #data = np.load(r"C:\Users\karim\PycharmProjects\HumanML3D\" + "kit_with_splits_2023.npz",allow_pickle=True)
        #normalized_poses =  np.asarray([xyz.reshape(-1, n_joint * 3) for xyz in data['kitmld_array']], dtype=object)
        h3d_data = np.load(r"C:\Users\karim\PycharmProjects\HumanML3D\all_humanML3D.npz",allow_pickle=True)
        data = np.asarray([xyz.reshape(-1, n_joint * 3) for xyz in h3d_data['kitmld_array']], dtype=object)
        descriptions = h3d_data["descriptions"] #[S for S in data["descriptions"]]

        # avoid skipping some frames due to memory limit
        matplotlib.rcParams['animation.embed_limit'] = 2 ** 128

    Tmax = 700*4
    ani = SubplotAnimation(data,frames=range(len(data[sample])), use_kps_3d=range(n_joint),
                           down_sample_factor=2,sample=sample,pred="A person turns something right left to the ground".split(" "),
                           idxs=[7*i for i in range(9)],dataset_name=dataset_name)

    # """"save animation
    rc('animation', html='jshtml')
    name_file = f"test_sample_{sample}.gif"
    ani.save(name_file)
    print("Animation saved in "+name_file)#os.getcwd()+