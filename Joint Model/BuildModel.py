import tensorflow as tf
import numpy, random, math
from GaitCore import Core
import pandas as pd
import matplotlib.pyplot as plt

from Vicon.Vicon import Vicon
from Vicon.Markers import Markers

frames = {"Root": [Core.Point.Point(0, 14, 0),
                   Core.Point.Point(56, 0, 0),
                   Core.Point.Point(14, 63, 0),
                   Core.Point.Point(56, 63, 0)], "L_Foot": [Core.Point.Point(0, 0, 0),
                                                            Core.Point.Point(70, 0, 0),
                                                            Core.Point.Point(28, 70, 0),
                                                            Core.Point.Point(70, 63, 0)],
          "L_Tibia": [Core.Point.Point(0, 0, 0),
                      Core.Point.Point(0, 63, 0),
                      Core.Point.Point(70, 14, 0),
                      Core.Point.Point(35, 49, 0)], "L_Femur": [Core.Point.Point(0, 0, 0),
                                                                Core.Point.Point(70, 0, 0),
                                                                Core.Point.Point(0, 42, 0),
                                                                Core.Point.Point(70, 56, 0)],
          "R_Foot": [Core.Point.Point(0, 0, 0),
                     Core.Point.Point(56, 0, 0),
                     Core.Point.Point(0, 49, 0),
                     Core.Point.Point(42, 70, 0)], "R_Tibia": [Core.Point.Point(0, 0, 0),
                                                               Core.Point.Point(42, 0, 0),
                                                               Core.Point.Point(7, 49, 0),
                                                               Core.Point.Point(63, 70, 0)],
          "R_Femur": [Core.Point.Point(7, 0, 0),
                      Core.Point.Point(56, 0, 0),
                      Core.Point.Point(0, 70, 0),
                      Core.Point.Point(42, 49, 0)]}


framesKnee = {"back": [Core.Point.Point(0, 14, 0),
                   Core.Point.Point(56, 0, 0),
                   Core.Point.Point(14, 63, 0),
                   Core.Point.Point(56, 63, 0)],
          "shank": [Core.Point.Point(0, 0, 0),
                      Core.Point.Point(0, 63, 0),
                      Core.Point.Point(70, 14, 0),
                      Core.Point.Point(35, 49, 0)], "thigh": [Core.Point.Point(0, 0, 0),
                                                                Core.Point.Point(70, 0, 0),
                                                                Core.Point.Point(0, 42, 0),
                                                                Core.Point.Point(70, 56, 0)]}


Ms = []


def single_step(joints, parents, children, shuffle_bodies=False, shuffle_markers=False):
    global Markers
    joints = [n.decode() for n in joints]
    parents = [n.decode() for n in parents]
    children = [n.decode() for n in children]
    for j in range(len(joints)):
        for m in Markers:
            timestep_dat = []
            timestep_labels = []
            joint = m.get_joint(joints[j])
            parent = m.get_rigid_body(parents[j])
            child = m.get_rigid_body(children[j])
            for i in range(len(joint)):
                joint_timestep = joint[i]
                parent_timestep = [parent[n][i] for n in range(len(parent))]
                child_timestep = [child[n][i] for n in range(len(child))]
                timestep_labels = [float(n) for n in joint_timestep]
                parent_unpacked = []
                child_unpacked = []

                if shuffle_markers:
                    random.shuffle(parent_timestep)
                    random.shuffle(child_timestep)

                for k in parent_timestep:
                    parent_unpacked = parent_unpacked + [float(k.x), float(k.y), float(k.z)]

                for k in child_timestep:
                    child_unpacked = child_unpacked + [float(k.x), float(k.y), float(k.z)]

                n = [parent_unpacked, child_unpacked]
                if shuffle_bodies:
                    random.shuffle(n)

                timestep_dat = n[0] + n[1]

                lowest_x = avg([timestep_dat[n] for n in range(0, len(timestep_dat), 3)] + [timestep_labels[0]])
                lowest_y = avg([timestep_dat[n] for n in range(1, len(timestep_dat), 3)] + [timestep_labels[1]])
                lowest_z = avg([timestep_dat[n] for n in range(2, len(timestep_dat), 3)] + [timestep_labels[2]])

                for n in range(0, len(timestep_dat)-2, 3):
                    timestep_dat[n] -= lowest_x
                    timestep_dat[n+1] -= lowest_y
                    timestep_dat[n+2] -= lowest_z

                timestep_labels[0] -= lowest_x
                timestep_labels[1] -= lowest_y
                timestep_labels[2] -= lowest_z

                yield timestep_dat, timestep_labels


def rel_sequence_of_n(joints, parents, children, n):
    global Ms
    joints = [n.decode() for n in joints]
    parents = [n.decode() for n in parents]
    children = [n.decode() for n in children]
    for j in range(len(joints)):
        print(joints[j])
        for m in Ms:
            joint = m.get_joint(joints[j])
            for i in range(len(joint)-(n-1)):
                seq = []
                lowest = [0, 0, 0]
                for k in range(n):
                    p_cen = m.body_centroid(parents[j], i+k)
                    c_cen = m.body_centroid(children[j], i+k)
                    lab = joint[i+k]
                    lowest = p_cen
                    c_cen = [c_cen[l] - lowest[l] for l in range(3)]
                    seq.append(c_cen)

                label = joint[i + n - 1]
                label = [label[l] - lowest[l] for l in range(3)]
                seq = [[m/300 for m in n] for n in seq]
                label = [m/300 for m in label]

                rel_start = m.body_rel_body(children[j], parents[j], i)
                rel_end = m.body_rel_body(children[j], parents[j], i+n-1)

                rel_start = [rel_start.x, rel_start.y, rel_start.z]
                rel_end = [rel_end.x, rel_end.y, rel_end.z]

                if pythag_loss(numpy.asarray([rel_start]), numpy.asarray([rel_end])) >= 15:
                    yield seq, label


def rel_sequence_of_n_knee(joints, parents, children, n):
    global Ms
    joints = [n.decode() for n in joints]
    parents = [n.decode() for n in parents]
    children = [n.decode() for n in children]
    for j in range(len(joints)):
        for m in Ms:
            joint = m.get_joint(joints[j])
            for i in range(len(m.get_rigid_body(parents[j])[0])-(n-1)):
                seq = []
                lowest = [0, 0, 0]
                for k in range(n):
                    p_cen = m.body_centroid(parents[j], i+k)
                    c_cen = m.body_centroid(children[j], i+k)

                    lowest = p_cen
                    c_cen = [c_cen[l] - lowest[l] for l in range(3)]
                    seq.append(c_cen)

                label = joint[i + n - 1]
                label = [label[l] - lowest[l] for l in range(3)]
                seq = [[m/300 for m in n] for n in seq]
                label = [m/300 for m in label]

                yield seq, label


def avg(n):
    return sum(n)/len(n)


def pythag_loss(y_pred, y_true):
    return tf.norm(y_pred-y_true, axis=1)


def setup(sources):
    for source in sources:
        v = Vicon(source)
        Ms.append(v.get_markers())

    for m in Ms:
        m.smart_sort()
        m.auto_make_transform(framesKnee)
        # for j in [["r_hip", "Root", "R_Femur", True],
        #           ["l_hip", "Root", "L_Femur", True],
        #           ["r_knee", "R_Femur", "R_Tibia", False],
        #           ["l_knee", "L_Femur", "L_Tibia", False],
        #           ["r_ankle", "R_Tibia", "R_Foot", True],
        #           ["l_ankle", "L_Tibia", "L_Foot", True]]:
        for j in [["knee", "thigh", "shank", False]]:
            m.def_joint(j[0], j[1], j[2], j[3])
        m.calc_joints(verbose=True)
        m.save_joints()


model_name = "LSTM_Knee_exodata_mocap_len10_gen2"
seq_len = 10

# sources = ["C:\\Users\\alekj\\IdeaProjects\\AIM_Vicon\\Vicon\\Examples\\ExampleData\\subject_03 Cal 03.csv",
#            "C:\\Users\\alekj\\IdeaProjects\\AIM_Vicon\\Vicon\\Examples\\ExampleData\\subject_02 Cal 01.csv",
#            "C:\\Users\\alekj\\IdeaProjects\\AIM_Vicon\\Vicon\\Examples\\ExampleData\\subject_00 Cal 01.csv",
#            "C:\\Users\\alekj\\IdeaProjects\\AIM_Vicon\\Vicon\\Examples\\ExampleData\\subject_00 Cal 02.csv",
#            "C:\\Users\\alekj\\IdeaProjects\\AIM_Vicon\\Vicon\\Examples\\ExampleData\\subject_06 Cal 01.csv"]

sources = ["C:\\Users\\alekj\\IdeaProjects\\AIM_Vicon\\Vicon\\Examples\\ExampleData\\knee\\knee_center Cal 01.csv",
           "C:\\Users\\alekj\\IdeaProjects\\AIM_Vicon\\Vicon\\Examples\\ExampleData\\knee\\knee_center Cal 02.csv",
           "C:\\Users\\alekj\\IdeaProjects\\AIM_Vicon\\Vicon\\Examples\\ExampleData\\knee\\knee_center Cal 03.csv",
           "C:\\Users\\alekj\\IdeaProjects\\AIM_Vicon\\Vicon\\Examples\\ExampleData\\knee\\knee_center Cal 04.csv"]

# joints = [["r_hip", "Root", "R_Femur", True],
#           ["l_hip", "Root", "L_Femur", True],
#           ["r_knee", "R_Femur", "R_Tibia", False],
#           ["l_knee", "L_Femur", "L_Tibia", False],
#           ["r_ankle", "R_Tibia", "R_Foot", True],
#           ["l_ankle", "L_Tibia", "L_Foot", True]]

joints = [["knee", "thigh", "shank", False]]

setup(sources)


ds = tf.data.Dataset.from_generator(rel_sequence_of_n_knee,
                                    args=[[n[0] for n in joints],
                                          [n[1] for n in joints],
                                          [n[2] for n in joints],
                                          seq_len],
                                    output_types=(tf.float32, tf.float32),
                                    output_shapes=((seq_len, 3), (3,)))
ds_batched = ds.shuffle(100).batch(50)


print("Data!")

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(3, input_shape=(seq_len, 3), activation="tanh"),
    tf.keras.layers.LSTM(96, return_sequences=True),
    tf.keras.layers.LSTM(96, return_sequences=True),
    tf.keras.layers.LSTM(96),
    tf.keras.layers.Dense(96, activation="tanh"),
    tf.keras.layers.Dense(3)
], model_name)
model.summary()

print("Model!")

model.compile(optimizer=tf.optimizers.Adam(0.001), loss=pythag_loss)
history = model.fit(ds_batched, epochs=100, verbose=0)

hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
hist.tail()

plt.plot(history.history['loss'], label='loss')
plt.xlabel('Epoch')
plt.ylabel('Error')
plt.legend()
plt.grid(True)

plt.show()

model.save(model_name)
