import math

import tensorflow as tf
import matplotlib.pyplot as plt
import random, numpy
from Vicon.Vicon import Vicon
from GaitCore import Core
from Vicon.Markers import Markers

model_name = "DFF_AllJoints_NoNorm_NoShuffle_Gen3"

sources = ["C:\\Users\\alekj\\IdeaProjects\\AIM_Vicon\\Vicon\\Examples\\ExampleData\\subject_03 Cal 03.csv",
           "C:\\Users\\alekj\\IdeaProjects\\AIM_Vicon\\Vicon\\Examples\\ExampleData\\subject_02 Cal 01.csv",
           "C:\\Users\\alekj\\IdeaProjects\\AIM_Vicon\\Vicon\\Examples\\ExampleData\\subject_00 Cal 01.csv",
           "C:\\Users\\alekj\\IdeaProjects\\AIM_Vicon\\Vicon\\Examples\\ExampleData\\subject_00 Cal 02.csv",
           "C:\\Users\\alekj\\IdeaProjects\\AIM_Vicon\\Vicon\\Examples\\ExampleData\\subject_06 Cal 01.csv"]

# sources = ["C:\\Users\\alekj\\IdeaProjects\\AIM_Vicon\\Vicon\\Examples\\ExampleData\\knee\\knee_center Cal 01.csv",
#            "C:\\Users\\alekj\\IdeaProjects\\AIM_Vicon\\Vicon\\Examples\\ExampleData\\knee\\knee_center Cal 02.csv",
#            "C:\\Users\\alekj\\IdeaProjects\\AIM_Vicon\\Vicon\\Examples\\ExampleData\\knee\\knee_center Cal 03.csv",
#            "C:\\Users\\alekj\\IdeaProjects\\AIM_Vicon\\Vicon\\Examples\\ExampleData\\knee\\knee_center Cal 04.csv"]


joints = [["r_hip", "Root", "R_Femur", True],
          ["l_hip", "Root", "L_Femur", True],
          ["r_knee", "R_Femur", "R_Tibia", False],
          ["l_knee", "L_Femur", "L_Tibia", False],
          ["r_ankle", "R_Tibia", "R_Foot", True],
          ["l_ankle", "L_Tibia", "L_Foot", True]]

# joints = [["knee", "thigh", "shank", False]]

Ms = []

Vicons = []

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

shuffle_bodies = False
shuffle_markers = False

ind = ""


def single_step(joints, parents, children, shuffle_bodies=False, shuffle_markers=False, up_to=99999):
    global Markers, ind
    # joints = [n.decode() for n in joints]
    # parents = [n.decode() for n in parents]
    # children = [n.decode() for n in children]
    for j in range(len(joints)):
        print("Gathering data for joint " + joints[j])
        for l in range(len(Ms)):
            m = Ms[l]
            v = Vicons[l]
            timestep_dat = []
            timestep_labels = []
            joint = m.get_joint(joints[j])
            parent = m.get_rigid_body(parents[j])
            child = m.get_rigid_body(children[j])

            if joints[j] == "r_hip":
                angles = v.get_model_output().get_right_leg().hip.angle
            elif joints[j] == "l_hip":
                angles = v.get_model_output().get_left_leg().hip.angle
            elif joints[j] == "r_knee":
                angles = v.get_model_output().get_right_leg().knee.angle
            elif joints[j] == "l_knee":
                angles = v.get_model_output().get_left_leg().knee.angle
            elif joints[j] == "r_ankle":
                angles = v.get_model_output().get_right_leg().ankle.angle
            else:
                angles = v.get_model_output().get_left_leg().ankle.angle

            for i in range(min([len(joint), up_to])):
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


def setup(sources):
    for source in sources:
        v = Vicon(source)
        Vicons.append(v)
        Ms.append(v.get_markers())

    for m in Ms:
        m.smart_sort()
        m.auto_make_transform(frames)
        for j in [["r_hip", "Root", "R_Femur", True],
                  ["l_hip", "Root", "L_Femur", True],
                  ["r_knee", "R_Femur", "R_Tibia", False],
                  ["l_knee", "L_Femur", "L_Tibia", False],
                  ["r_ankle", "R_Tibia", "R_Foot", True],
                  ["l_ankle", "L_Tibia", "L_Foot", True]]:
        # for j in [["knee", "thigh", "shank", False]]:
            m.def_joint(j[0], j[1], j[2], j[3])
        m.calc_joints(verbose=True)
        m.save_joints()


def rel_sequence_of_n(joints, parents, children, n, up_to=999999):
    global Ms, ind
    #joints = [n.decode() for n in joints]
    #parents = [n.decode() for n in parents]
    #children = [n.decode() for n in children]
    for j in range(len(joints)):
        for m_ind in range(len(Ms)):
            m = Ms[m_ind]
            v = Vicons[m_ind]
            joint = m.get_joint(joints[j])
            for i in range(min([len(joint)-(n-1), up_to])):
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

                yield seq, label


def rel_sequence_of_n_knee(joints, parents, children, n):
    global Ms
    joints = [n for n in joints]
    parents = [n for n in parents]
    children = [n for n in children]
    for j in range(len(joints)):
        for m in Ms:
            for i in range(len(m.get_rigid_body(parents[j])[0])-(n-1)):
                seq = []
                lowest = [0, 0, 0]
                for k in range(n):
                    p_cen = m.body_centroid(parents[j], i+k)
                    c_cen = m.body_centroid(children[j], i+k)

                    lowest = p_cen
                    c_cen = [c_cen[l] - lowest[l] for l in range(3)]
                    seq.append(c_cen)

                label = m.get_marker("knee")[i + n - 1]
                label = [label.x, label.y, label.z]
                label = [label[l] - lowest[l] for l in range(3)]
                seq = [[m/300 for m in n] for n in seq]
                label = [m/300 for m in label]

                yield seq, label


def rand_from_single_step(buff_size):
    buff = []
    js = [n[0] for n in joints]
    ps = [n[1] for n in joints]
    cs = [n[2] for n in joints]
    i = 0
    for n in single_step(js, ps, cs, shuffle_bodies, shuffle_markers):
        buff.append(n)
        i += 1
        if i >= buff_size:
            break

    return random.choice(buff)


def pythag_loss(y_pred, y_true):
    return tf.norm(y_pred-y_true, axis=1)


def view_single():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    dat, lab = rand_from_single_step(1000)
    ax.scatter([dat[3*n] for n in range(8)],
                [dat[3*n + 1] for n in range(8)],
                [dat[3*n + 2] for n in range(8)],
                c="b")
    ax.scatter([lab[0]], [lab[1]], [lab[2]], c="g")

    print(dat)
    guess = model.predict([dat])[0]
    ax.scatter(guess[0], guess[1], guess[2], c="r")
    plt.title(model_name + " " + ind)
    plt.show()


def dist_from_origin(points):
    return sum([math.sqrt(points[n]**2 + points[n+1]**2 + points[n+2]**2) for n in range(len(points)-2)])/len(points)


def loss_by_dist_from_origin():
    js = [n[0] for n in joints]
    ps = [n[1] for n in joints]
    cs = [n[2] for n in joints]
    loss = []
    dist = []
    for points, center in single_step(js, ps, cs, shuffle_bodies, shuffle_markers, up_to=500):
        lowest_x = avg([points[n] for n in range(0, len(points), 3)] + [center[0]])
        lowest_y = avg([points[n] for n in range(1, len(points), 3)] + [center[1]])
        lowest_z = avg([points[n] for n in range(2, len(points), 3)] + [center[2]])

        for n in range(0, len(points)-2, 3):
            points[n] -= lowest_x
            points[n+1] -= lowest_y
            points[n+2] -= lowest_z

        center[0] -= lowest_x
        center[1] -= lowest_y
        center[2] -= lowest_z

        guess = model.predict([points])[0]

        center[0] += lowest_x
        center[1] += lowest_y
        center[2] += lowest_z

        guess[0] += lowest_x
        guess[1] += lowest_y
        guess[2] += lowest_z

        dist.append(dist_from_origin(points))
        loss.append(pythag_loss(numpy.asarray([guess]), [center]).numpy()[0])

    plt.scatter(dist, loss)
    plt.xlabel("Average Distance From Origin As Seen By Model (mm)")
    plt.ylabel("Loss (mm)")
    plt.title("Loss by Average Distance From Origin")
    plt.show()


def loss_by_joint_angle():
    js = [n[0] for n in joints]
    ps = [n[1] for n in joints]
    cs = [n[2] for n in joints]
    loss = []
    ang = []
    loss_by_joint_type = {"r_hip":[], "l_hip":[], "r_knee":[], "l_knee":[], "r_ankle":[], "l_ankle":[]}
    for points, center, angl in rel_sequence_of_n(js, ps, cs, 2, up_to=500):
        # fact = max([max(n) for n in points])
        # points = [[m/fact for m in n] for n in points]
        # center = [n/fact for n in center]

        guess = model.predict([points])[0]

        # guess = [n * fact for n in guess]
        # center = [n * fact for n in center]

        loss.append(pythag_loss(numpy.asarray([guess]), [center]).numpy()[0])
        ang.append(norm2([angl.x, angl.y, angl.z]))
        loss_by_joint_type[ind[0]].append(pythag_loss(numpy.asarray([guess]), [center]).numpy()[0])

    plt.scatter(ang, loss)
    plt.xlabel("Angle of Joint")
    plt.ylabel("Loss (mm)")
    plt.title("Loss by Joint Angle")
    for i in loss_by_joint_type:
        print(i)
        print(avg(loss_by_joint_type[i]))
    plt.show()


def knee_metrics():
    js = [n[0] for n in joints]
    ps = [n[1] for n in joints]
    cs = [n[2] for n in joints]
    sLoss = []
    for seq, label in rel_sequence_of_n_knee(js, ps, cs, 10):
        guess = model.predict([seq])[0]
        sLoss.append(pythag_loss(numpy.asarray([guess]), numpy.asarray([label]))[0].numpy() * 300)

    print(sum(sLoss)/len(sLoss))
    print(max(sLoss))
    print(min(sLoss))
    print(len(sLoss))


def mocap_metrics():
    js = [n[0] for n in joints]
    ps = [n[1] for n in joints]
    cs = [n[2] for n in joints]
    loss = []
    for points, center in rel_sequence_of_n(js, ps, cs, 10, up_to=1000):
        guess = model.predict([points])[0]
        guess = guess[len(guess)-1]
        loss.append(pythag_loss(numpy.asarray([guess]), [center]).numpy()[0] * 300)

    print(sum(loss)/len(loss))
    print(max(loss))
    print(min(loss))
    print(len(loss))


def single_step_metrics():
    js = [n[0] for n in joints]
    ps = [n[1] for n in joints]
    cs = [n[2] for n in joints]
    loss = []
    for dat, lab in single_step(js, ps, cs, shuffle_bodies, shuffle_markers, up_to=1000):
        guess = model.predict([dat])[0]
        loss.append(pythag_loss(numpy.asarray([guess]), [lab]).numpy()[0])

    print(sum(loss)/len(loss))
    print(max(loss))
    print(min(loss))
    print(len(loss))


def avg(n):
    return sum(n)/len(n)

def norm2(n):
    return math.sqrt(sum([x**2 for x in n]))

custom_objects = {"pythag_loss": pythag_loss}

with tf.keras.utils.custom_object_scope(custom_objects):
    model = tf.keras.models.load_model(model_name)
    model.summary()

setup(sources)

single_step_metrics()
