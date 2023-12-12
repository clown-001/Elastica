import numpy as np
from elastica import *
from examples.RodContactCase.post_processing import (
    plot_video_with_surface,
    plot_velocity,
    plot_link_writhe_twist,
)

from matplotlib import pyplot as plt
from matplotlib.colors import to_rgb
from mpl_toolkits.mplot3d import proj3d, Axes3D
from tqdm import tqdm

from typing import Dict, Sequence

class Loopcase(
    BaseSystemCollection,
    Constraints,
    Connections,
    Forcing,
    CallBacks,
    Damping,
):
    pass


loop_sim = Loopcase()

# Simulation parameters
number_of_rotations = 20
time_start_twist = 0
time_twist = 100
time_compression = 250
time_tension = 250
time_intev = 100
final_time = time_compression + time_tension + time_twist + time_intev
base_length = 1.2
n_elem = 100

dt = 0.0025 * base_length / n_elem  # 1E-2
total_steps = int(final_time / dt)
time_step = np.float64(final_time / total_steps)
rendering_fps = 20
step_skip = int(1.0 / (rendering_fps * time_step))

# Rest of the rod parameters and construct rod
base_radius = 0.0075
base_area = np.pi * base_radius ** 2
volume = base_area * base_length
mass = 1
density = mass / volume
nu = 2.0 / density / base_area
E = 1e6
poisson_ratio = 0.5
shear_modulus = E / (poisson_ratio + 1.0)
slack = 0.8
direction = np.array([0.0, 1.0, 0.0])
normal = np.array([0.0, 0.0, 1.0])
start = np.zeros(
    3,
)

sherable_rod = CosseratRod.straight_rod(

    n_elem,
    start,
    direction,
    normal,
    base_length,
    base_radius,
    density,
    youngs_modulus=E,
    shear_modulus=shear_modulus,
)




loop_sim.append(sherable_rod)

# Add damping
loop_sim.dampen(sherable_rod).using(
    AnalyticalLinearDamper,
    damping_constant=nu,
    time_step=dt,
)

# boundary condition
from elastica._rotations import _get_rotation_matrix


class LoopBC(ConstraintBase):
    
    def __init__(
        self,
        position_start,
        position_end,
        director_start,
        director_end,
        twisting_time,        
        number_of_rotations,
        slack,
        compression_time,
        tension_time,
        intev_time,
        **kwargs,
    ):

        super().__init__(**kwargs)
        self.twisting_time = twisting_time
        self.compression_time = compression_time
        self.tension_time = tension_time
        self.intev_time = intev_time

        theta = 2.0 * number_of_rotations * np.pi

        angel_vel_scalar = theta / self.twisting_time
        shrink_vel_scalar = slack / (self.compression_time)
        stretch_vel_scalar = slack / (self.tension_time)

        direction = (position_end - position_start) / np.linalg.norm(
            position_end - position_start
        )


        self.final_start_position = position_start + slack / direction
        self.final_end_position = position_end - slack / direction

        self.ang_vel = angel_vel_scalar * direction /2
        self.shrink_vel = shrink_vel_scalar * direction /2
        self.stretch_vel = stretch_vel_scalar * direction /2

        axis_of_rotation_in_material_frame = director_end @ direction
        axis_of_rotation_in_material_frame /= np.linalg.norm(
            axis_of_rotation_in_material_frame
        )

        self.final_start_directors = (
            _get_rotation_matrix(theta, direction.reshape(3, 1)).reshape(3, 3)
            @ director_start
        )  # rotation_matrix wants vectors 3,1
        self.final_end_directors = (
            _get_rotation_matrix(-theta, direction.reshape(3, 1)).reshape(3, 3)
            @ director_end
        )  # rotation_matrix wants vectors 3,1
        self.ang_vel = angel_vel_scalar * axis_of_rotation_in_material_frame

        self.position_start = position_start
        self.director_start = director_start



    def constrain_values(
        self, rod, time: float
    ) -> None:
        if time > self.compression_time + self.tension_time + self.twisting_time+self.intev_time:

            rod.position_collection[..., 0] = self.position_start
            rod.position_collection[0, -1] = 0.0
            rod.position_collection[2, -1] = 0.0


            


    def constrain_rates(
        self, rod, time: float
    ) -> None:

        if time > self.compression_time + self.tension_time + self.twisting_time+self.intev_time:
            rod.velocity_collection[..., 0] = 0.0
            rod.omega_collection[..., 0] = 0.0

            rod.velocity_collection[..., -1] = 0.0
            rod.omega_collection[..., -1] = 0.0

        elif time > self.compression_time + self.twisting_time+self.intev_time:
            rod.velocity_collection[..., 0] = -self.shrink_vel
            rod.omega_collection[..., 0] = 0.0

            rod.velocity_collection[..., -1] = self.shrink_vel
            rod.omega_collection[..., -1] = 0.0            
        elif time > self.twisting_time + self.intev_time:
            rod.velocity_collection[..., 0] = self.shrink_vel
            rod.omega_collection[..., 0] = 0.0

            rod.velocity_collection[..., -1] = -self.shrink_vel
            rod.omega_collection[..., -1] = 0.0            

        elif time > self.twisting_time:
            rod.velocity_collection[..., 0] = 0.0
            rod.omega_collection[..., 0] = 0.0

            rod.velocity_collection[..., -1] = 0.0
            rod.omega_collection[..., -1] = 0.0

        else :
            rod.velocity_collection[..., 0] = 0.0
            rod.omega_collection[..., 0] = self.ang_vel

            rod.velocity_collection[..., -1] = 0.0
            rod.omega_collection[..., -1] = -self.ang_vel

loop_sim.constrain(sherable_rod).using(
    LoopBC,
    constrained_position_idx=(0, -1),
    constrained_director_idx=(0, -1),
    twisting_time=time_twist,
    slack=slack,
    number_of_rotations=number_of_rotations,
    compression_time=time_compression,
    tension_time=time_tension,
    intev_time = time_intev,
)

# Add self contact to prevent penetration
loop_sim.connect(sherable_rod, sherable_rod).using(SelfContact, k=1e4, nu=10)

# Add callback functions for plotting position of the rod later on
class RodCallBack(CallBackBaseClass):
    """ """

    def __init__(self, step_skip: int, callback_params: dict):
        CallBackBaseClass.__init__(self)
        self.every = step_skip
        self.callback_params = callback_params

    def make_callback(self, system, time, current_step: int):
        if current_step % self.every == 0:
            
            self.callback_params["time"].append(time)
            self.callback_params["step"].append(current_step)

            self.callback_params["position0"].append(system.position_collection[0,...].copy())
            self.callback_params["position1"].append(system.position_collection[1,...].copy())
            self.callback_params["position2"].append(system.position_collection[2,...].copy())
            self.callback_params["position"].append(system.position_collection.copy())
            self.callback_params["omega"].append(system.omega_collection[1,-1].copy())

            '''
            self.callback_params["directors"].append(system.director_collection[...,2,0].copy())

            '''
            self.callback_params["radius"].append(system.radius.copy())
            self.callback_params["com"].append(system.compute_position_center_of_mass())
            self.callback_params["com_velocity"].append(
                system.compute_velocity_center_of_mass()
            )
            '''

            self.callback_params["internal_forces"].append(system.internal_forces[1,0].copy())
            '''
            self.callback_params["internal_torques"].append(system.internal_torques[...,-1].copy())
            '''
            self.callback_params["external_forces"].append(system.external_forces[1,0].copy())

            total_energy = (
                system.compute_translational_energy()
                + system.compute_rotational_energy()
                + system.compute_bending_energy()
                + system.compute_shear_energy()
            )
            self.callback_params["total_energy"].append(total_energy)
            '''
            self.callback_params["directors"].append(system.director_collection.copy())
            '''
'''
            return


post_processing_dict = defaultdict(list)  # list which collected data will be append
# set the diagnostics for rod and collect data
loop_sim.collect_diagnostics(sherable_rod).using(
    RodCallBack,
    step_skip=step_skip,
    callback_params=post_processing_dict,
)

# finalize simulation
loop_sim.finalize()

# Run the simulation
time_stepper = PositionVerlet()
integrate(time_stepper, loop_sim, final_time, total_steps)

# plotting the videos
filename_video = "loop3.mp4"
plot_video_with_surface(
    [post_processing_dict],
    video_name=filename_video,
    fps=rendering_fps,
    step=1,
    vis3D=True,
    vis2D=True,
    x_limits=[-0.5, 0.5],
    y_limits=[-0.1, 1.1],
    z_limits=[-0.5, 0.5],
)
'''
# Compute topological quantities
time = np.array(post_processing_dict["time"])
position_history = np.array(post_processing_dict["position"])
radius_history = np.array(post_processing_dict["radius"])
director_history = np.array(post_processing_dict["directors"])




# Compute twist density
theta = 2.0 * number_of_rotations * np.pi
angel_vel_scalar = theta / time_twist

twist_time_interval_start_idx = np.where(time > 0)[0][0]
twist_time_interval_end_idx = np.where(time < (time_twist))[0][-1]

twist_density = (
    (time[twist_time_interval_start_idx:twist_time_interval_end_idx] )
    * angel_vel_scalar
    * base_radius
)

# Compute link-writhe-twist
normal_history = director_history[:, 0, :, :]
segment_length = 10 * base_length

type_of_additional_segment = "next_tangent"

total_twist, local_twist = compute_twist(position_history, normal_history)

total_link = compute_link(
    position_history,
    normal_history,
    radius_history,
    segment_length,
    type_of_additional_segment,
)

total_writhe = compute_writhe(
    position_history, segment_length, type_of_additional_segment
)

# Plot link-writhe-twist
plot_link_writhe_twist(
    twist_density,
    total_twist[twist_time_interval_start_idx:twist_time_interval_end_idx],
    total_writhe[twist_time_interval_start_idx:twist_time_interval_end_idx],
    total_link[twist_time_interval_start_idx:twist_time_interval_end_idx],
)

def plot_torque(
    post_processing_dict: dict,
    filename="torque.png",
    SAVE_FIGURE=1,

):
    displacement = np.array(post_processing_dict["position"])
    rod_external_torque = np.array(post_processing_dict["external_torque"])

    fig = plt.figure(figsize=(12, 10), frameon=True, dpi=150)
    axs = []
    axs.append(plt.subplot2grid((4, 1), (0, 0)))
    axs.append(plt.subplot2grid((4, 1), (1, 0)))
    axs.append(plt.subplot2grid((4, 1), (2, 0)))
    axs[0].plot(time[:], rod_external_torque[:], linewidth=3, label="internal_forces")
 
    axs[0].set_ylabel("x torque", fontsize=20)

    axs[1].plot(time[:],rod_external_torque[:],linewidth=3,)



    axs[1].set_ylabel("y torque", fontsize=20)
    axs[2].plot(time[:],rod_external_torque[:],linewidth=3,)

    axs[2].set_ylabel("z torque", fontsize=20)

    

    plt.tight_layout()
    # fig.align_ylabels()
    fig.legend(prop={"size": 20})
    # fig.savefig(filename)
    plt.show()
    plt.close(plt.gcf())

    if SAVE_FIGURE:
        fig.savefig(filename)
        # plt.savefig(filename)

def plot_force(
    post_processing_dict: dict,
    filename="force.png",
    SAVE_FIGURE=1,

):
    time = np.array(post_processing_dict["time"])
    position_history = np.array(post_processing_dict["position"])
    rod_internal_forces = np.array(post_processing_dict["internal_forces"])

    fig = plt.figure(figsize=(12, 10), frameon=True, dpi=150)
    axs = []
    axs.append(plt.subplot2grid((4, 1), (0, 0)))


    axs[0].plot(position_history[:],rod_internal_forces[:],linewidth=3,)



    axs[0].set_ylabel("y force", fontsize=20)
    

    plt.tight_layout()
    # fig.align_ylabels()
    fig.legend(prop={"size": 20})
    # fig.savefig(filename)
    plt.show()
    plt.close(plt.gcf())

    if SAVE_FIGURE:
        fig.savefig(filename)
        # plt.savefig(filename)

plot_force(
    post_processing_dict,
    filename='FORCE',
    SAVE_FIGURE=1,
    )
'''    
# Save simulation data

position0_history = np.array(post_processing_dict["position0"])
position1_history = np.array(post_processing_dict["position1"])
position2_history = np.array(post_processing_dict["position2"])
omega_history = np.array(post_processing_dict["omega"])
'''
director_history = np.array(post_processing_dict["directors"])

rod_internal_forces = np.array(post_processing_dict["internal_forces"])
'''
rod_internal_torques = np.array(post_processing_dict["internal_torques"])
'''
np.savetxt(
    "C:/Users/tjh/Desktop/data/director.txt",
    director_history,
    fmt="%f"


)
'''
np.savetxt(
    "C:/Users/301/Desktop/data/omega.txt",
    omega_history,
    fmt="%f"


)
np.savetxt(
    "C:/Users/301/Desktop/data/position0.txt",
    position0_history,
    fmt="%f"


)
np.savetxt(
    "C:/Users/301/Desktop/data/position1.txt",
    position1_history,
    fmt="%f"


)
np.savetxt(
    "C:/Users/301/Desktop/data/position2.txt",
    position2_history,
    fmt="%f"


)


'''
np.savetxt(
    "C:/Users/tjh/Desktop/data/force.txt",
    rod_internal_forces,
    fmt="%f"

)
'''
np.savetxt(
    "C:/Users/301/Desktop/data/torques.txt",
    rod_internal_torques,
    fmt="%f"

)
