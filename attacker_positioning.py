import time
import numpy as np
from dimensions import *
from colors import *
import cv2

def is_inside_interval(value, x_min, x_max):
    if value>x_min and value<x_max:
        return True
    else: 
        return False

def map(value, in_min, in_max, out_min, out_max):
    if value <= in_min:
        return out_min
    elif value >= in_max:
        return out_max
    else:
        return out_min + (value-in_min)*(out_max-out_min)/(in_max-in_min)

def draw_env_limits(img, env_limits, env_color):
    img_center_x, img_center_y = img.shape[1]/2, img.shape[0]/2
    xmin, xmax, ymin, ymax = env_limits
    pt1 = int(xmin/10 + img_center_x), int(ymin/10 + img_center_y)
    pt2 = int(xmax/10 + img_center_x), int(ymax/10 + img_center_y)
    cv2.rectangle(img, pt1, pt2, env_color, 1)

def visualize_with_opencv(current_command, current_likelihood, average_likelihood, weights, particles, enemies, field, gk_area, particles_workspace, frame_duration=0):
    img_height, img_width = int(field_width/10 + 2*boundary_width/10), int(field_length/10 + 2*boundary_width/10)
    img = np.zeros((img_height, img_width, 3), dtype=np.uint8)
    
    for weight, particle in zip(weights, particles):
        x, y = int(particle[0]/10+img_width/2), int(particle[1]/10+img_height/2)
        radius = int(map(weight, 0.5, 0.95, 2, 20))
        cv2.circle(img, (x, y), radius, BLUE, 1)
    
    for enemy in enemies:
        x, y = int(enemy[0]/10+img_width/2), int(enemy[1]/10+img_height/2)
        radius = 18
        cv2.circle(img, (x, y), radius, RED, -1)
    
    average_particle = get_average_particle(weights, particles)
    x, y = int(average_particle[0]/10+img_width/2), int(average_particle[1]/10+img_height/2)
    cv2.circle(img, (x, y), 9, GREEN, -1)
    
    x, y = int(current_command[0]/10+img_width/2), int(current_command[1]/10+img_height/2)
    cv2.circle(img, (x, y), 9, LIGHT_BLUE, -1)
    
    draw_env_limits(img, field, WHITE)
    draw_env_limits(img, gk_area, WHITE)
    draw_env_limits(img, particles_workspace, RED)
    
    cv2.putText(img, f'Current Likelihood: {current_likelihood:0.3f}', (40, 50), 1, 1.0, WHITE, 1)
    cv2.putText(img, f'Best Likelihood: {average_likelihood:0.3f}', (40, 70), 1, 1.0, WHITE, 1)
    
    cv2.imshow('test', img)
    
    key = cv2.waitKey(frame_duration) & 0xFF
    if key == ord('q'):
        cv2.destroyAllWindows()
        quit()
    elif key==ord('b'):
        return True
    else:
        return False

def get_average_particle(weights, particles):
    # Compute the weighted sum of the particles
    average_particle = np.sum(weights[:, np.newaxis] * particles, axis=0)
    
    return average_particle

def is_out_of_environment(particle, env_limits):
    '''
    Check if particle is out of field boundaries
    
    param: particle position and environment limits
    return: True if particle is out of env boundaries
    '''
    xmin, xmax, ymin, ymax = env_limits
    if particle[0] < xmin or \
       particle[0] > xmax or \
       particle[1] < ymin or \
       particle[1] > ymax:
        return True
    else:
        return False

def is_inside_zone(particle, zone_limits):
    '''
    Check if particle is out of field boundaries
    
    param: particle position and environment limits
    return: True if particle is out of env boundaries
    '''
    xmin, xmax, ymin, ymax = zone_limits
    if particle[0] > xmin and \
       particle[0] < xmax and \
       particle[1] > ymin and \
       particle[1] < ymax:
        return True
    else:
        return False

def generate_random_particle(env_limits, prohibited_zone_limits):
    env_xmin, env_xmax, env_ymin, env_ymax = env_limits
    zone_xmin, zone_xmax, zone_ymin, zone_ymax = prohibited_zone_limits
    x = np.random.uniform(env_xmin, env_xmax)
    y = np.random.uniform(env_ymin, env_ymax)
    particle = np.array([x, y])
    
    while is_out_of_environment(particle, env_limits) or \
            is_inside_zone(particle, prohibited_zone_limits):
        x = np.random.uniform(env_xmin, env_xmax)
        y = np.random.uniform(env_ymin, env_ymax)
        particle = np.array([x, y])
    
    return particle

def initialize_particles_uniform(n_particles, env_limits, prohibited_zone_limits):
    env_xmin, env_xmax, env_ymin, env_ymax = env_limits
    zone_xmin, zone_xmax, zone_ymin, zone_ymax = prohibited_zone_limits
    
    # Initialize particles with uniform weight distribution
    particles = np.zeros((n_particles, 2))
    weight = 1.0/n_particles
    for i in range(n_particles):
        x = np.random.uniform(env_xmin, env_xmax)
        y = np.random.uniform(env_ymin, env_ymax)
        particle = np.array([x, y])
        
        while is_out_of_environment(particle, env_limits) or \
              is_inside_zone(particle, prohibited_zone_limits):
            x = np.random.uniform(env_xmin, env_xmax)
            y = np.random.uniform(env_ymin, env_ymax)
            particle = np.array([x, y])
        
        particles[i] = particle
    
    weights = weight*np.ones(n_particles)
    
    return weights, particles

def normalize_weights(weights):
    if np.sum(weights) < EPS:
        n_particles = len(weights)
        return np.ones_like(weights)/n_particles
    else:
        return weights/np.sum(weights)

def generate_goalkeeper(gk_area_limits, goal_width):
    zone_xmin, zone_xmax, zone_ymin, zone_ymax = gk_area_limits
    
    # Initialize goalkeeper
    x = np.random.uniform(zone_xmax - 2*robot_diameter, zone_xmax)
    y = np.random.uniform(-goal_width/2, goal_width/2)
    gk = np.array([x, y])
                
    return gk 

def generate_enemies(n_enemies, env_limits, gk_area_limits, has_goalkeeper):
    env_xmin, env_xmax, env_ymin, env_ymax = env_limits
    
    # Initialize enemies
    enemies = np.zeros((n_enemies, 2))
    
    if has_goalkeeper:
        enemies[0] = generate_goalkeeper(gk_area_limits=GK_AREA,
                                         goal_width=goal_width)

    for i in range(int(has_goalkeeper), n_enemies):
        x = np.random.uniform(env_xmin, env_xmax)
        y = np.random.uniform(env_ymin, env_ymax)
        enemy = np.array([x, y])
        
        while is_out_of_environment(enemy, env_limits) or \
              is_inside_zone(enemy, gk_area_limits):
            x = np.random.uniform(env_xmin, env_xmax)
            y = np.random.uniform(env_ymin, env_ymax)
            enemy = np.array([x, y])
        
        enemies[i] = enemy
        
    return enemies

def get_dist(p1, p2):
    if p2.ndim>1:
        return np.linalg.norm(p2-p1, axis=1)
    else:  
        return np.linalg.norm(p2-p1)

def get_distance_to_goal_center(particle, field_limits):
    field_xmin, field_xmax, field_ymin, field_ymax = field_limits
    goal_center = np.array([field_xmax, 0])
    distance = get_dist(particle, goal_center)
    return distance

def compute_observation_from_distance_to_goal(measurement, ideal_distance=goalkeeper_area_length + robot_diameter / 2, alpha=2):
    normalize_distance = abs(measurement - ideal_distance) / ideal_distance
    p_z = np.exp(-alpha * normalize_distance)
    return p_z

def get_distance_to_closest_enemy(particle, enemies):
    distance = np.min(get_dist(particle, enemies))
    return distance

def compute_observation_from_closest_enemy_distance(measurement, worst_distance=robot_diameter, alpha=0.5):
    if measurement<robot_diameter:
        return 0
    else:
        normalized_distance = abs(measurement - worst_distance) / worst_distance
        interception_likelihood = np.exp(-alpha * normalized_distance)
        p_z = 1 - interception_likelihood
        return p_z

def get_angle_to_goal(particle, goal_width, field_limits):
    field_xmin, field_xmax, field_ymin, field_ymax = field_limits
    goal_upper_post = np.array([field_xmax, goal_width/2])
    goal_lower_post = np.array([field_xmax, -goal_width/2])
    v1 = goal_upper_post - particle
    v2 = goal_lower_post - particle
    prod_norm = (np.linalg.norm(v1)*np.linalg.norm(v2))
    if prod_norm==0: 
        return 0
    else:
        angle_cos = np.dot(v2,v1)/prod_norm
        angle_deg = np.rad2deg(np.arccos(angle_cos))
        return angle_deg

def compute_observation_from_angle_to_goal(measurement, best_angle, alpha=0.15):
    if measurement<10:
        return 0
    else:
        normalized_angle_diff = abs(measurement - best_angle) / best_angle
        angle_likelihood = np.exp(-alpha * normalized_angle_diff)
        p_z = angle_likelihood
        return p_z    

def get_free_space_on_goal(particle, enemies, goal_width, field_limits):
    goal_center = np.array([field_limits[1], 0])
    goal_segment = np.array([-goal_width/2, goal_width/2])
    r = robot_diameter/2
    shadow_segments = []
    for enemy in enemies:
        if (np.dot(enemy-particle, goal_center-particle)>=0):
            # 1) Compute distances to intersection points
            l = np.sqrt(r**2 + get_dist(particle, enemy)**2)
            
            # 2) Compute intersection points' coordinates w.r.t. the particle
            #    (x1, y1) and (x2, y2) are actually the vector: intersection point - particle
            #    we use this vector because it simplifies our desired calculations
            x1 = (enemy[0]+r*enemy[1]/l)/(1+r**2/l**2)
            y1 = (enemy[1]-r*enemy[0]/l)/(1+r**2/l**2)
            x2 = (enemy[0]-r*enemy[1]/l)/(1+r**2/l**2)
            y2 = (enemy[1]+r*enemy[0]/l)/(1+r**2/l**2)
            
            # 3) Compute y coordiantes of the projections:
            #    particle->intersection point->goal line
            y1_proj = particle[1] + y1*(goal_center[0]-particle[0])/x1
            y2_proj = particle[1] + y2*(goal_center[0]-particle[0])/x2
            should_add_to_segments_list = is_inside_interval(y1_proj, np.min(goal_segment), np.max(goal_segment)) or \
                                          is_inside_interval(y2_proj, np.min(goal_segment), np.max(goal_segment))
            if should_add_to_segments_list:
                # 4) Add coordinates to segments' list
                segment_bounds = np.array([y1_proj, y2_proj])
                shadow_segments.append(segment_bounds)
    
    # 5) Check if shadows exist
    if len(shadow_segments)==0:
        return np.max(goal_segment)-np.min(goal_segment)
    
    # 6) Sort segments by their starting points
    shadow_segments = np.sort(shadow_segments, axis=0)
    
    # Merge the intervals of the other segments
    merged_intervals = []
    current_start, current_end = shadow_segments[0]

    for start, end in shadow_segments[1:]:
        if start <= current_end:
            current_end = max(current_end, end)
        else:
            merged_intervals.append([current_start, current_end])
            current_start, current_end = start, end
    merged_intervals.append([current_start, current_end])
    
    # Calculate the free spaces within the first segment
    free_spaces = []
    first_start, first_end = np.min(goal_segment), np.max(goal_segment)

    # Adjust merged intervals to lie within the first segment's boundaries
    adjusted_intervals = [
        [max(first_start, interval[0]), min(first_end, interval[1])]
        for interval in merged_intervals
        if max(first_start, interval[0]) < min(first_end, interval[1])
    ]

    # Check for free space before the first adjusted interval
    if adjusted_intervals[0][0] > first_start:
        free_spaces.append([first_start, adjusted_intervals[0][0]])

    # Check for free spaces between adjusted intervals
    for i in range(1, len(adjusted_intervals)):
        if adjusted_intervals[i][0] > adjusted_intervals[i-1][1]:
            free_spaces.append([adjusted_intervals[i-1][1], adjusted_intervals[i][0]])

    # Check for free space after the last adjusted interval
    if adjusted_intervals[-1][1] < first_end:
        free_spaces.append([adjusted_intervals[-1][1], first_end])
    
    # Calculate the lengths of the free spaces
    free_space_lengths = [end - start for start, end in free_spaces]

    # Return the length of the longest free space
    if len(free_spaces)>1:
        return np.sum(free_space_lengths)
    else:
        return 0

def compute_observation_from_free_space_on_goal(measurement, ideal_space=goal_width, alpha=0.1):
    normalize_space = abs(measurement - ideal_space) / ideal_space
    p_z = np.exp(-alpha * normalize_space)
    return p_z

def cumulative_sum(weights):
    """
    Compute cumulative sum of a list of scalar weights

    :param weights: list with weights
    :return: list containing cumulative weights, length equal to length input
    """
    return np.cumsum(weights).tolist()

def systematic_resample(weights, particles, n_particles, delta, search_factor, env_limits, prohibited_zone_limits):
    """
    Loop over cumulative sum once hence particles should keep same order (however some disappear, other are
    replicated). Variance on number of times a particle will be selected lower than with stratified resampling.

    Computational complexity: O(N)

    :param samples: Samples that must be resampled.
    :param N: Number of samples that must be generated.
    :return: Resampled weighted particles.
    """
    # Compute cumulative sum
    Q = cumulative_sum(weights)

    # Only draw one sample
    u0 = np.random.uniform(1e-10, Q[-1] / n_particles, 1)[0]

    # As long as the number of new samples is insufficient
    n = 0
    m = 0  # index first element
    new_samples = []
    while n < n_particles:

        # Compute u for current particle (deterministic given u0)
        u = u0 + float(n)*Q[-1] / n_particles

        # u increases every loop hence we only move from left to right while iterating Q

        # Get first sample for which cumulative sum is above u
        while Q[m] < u:
            m += 1

        # Search factor
        if np.random.uniform(0, 1) < search_factor:
            new_sample = generate_random_particle(env_limits, prohibited_zone_limits)
        else:
            # Add state sample (uniform weights)
            rnd = np.random.uniform(-1, 1, 2)
            new_sample = particles[m] + (1-weights[m]/Q[-1])*delta*rnd
            
        new_samples.append(new_sample)

        # Added another sample
        n += 1

    # Reset weights
    weights = np.ones(n_particles)/n_particles

    # Return new samples
    return weights, np.array(new_samples)

def get_highest_angle_to_shoot(gk_area_limits, goal_width, field_limits):
    gk_area_xmin = gk_area_limits[0]
    closest_position_to_shoot = np.array([gk_area_xmin-robot_diameter, 0])  
    return get_angle_to_goal(closest_position_to_shoot, goal_width, field_limits)

def compute_particle_likelihood(particle, enemies, goal_width, best_shooting_angle, GK_AREA_WITH_MARGINS, FIELD):
    distance_to_goal = get_distance_to_goal_center(particle, FIELD)
    distance_to_closest_enemy = get_distance_to_closest_enemy(particle, enemies)
    angle_to_goal = get_angle_to_goal(particle, goal_width, FIELD)
    #best_free_space_on_goal = get_free_space_on_goal(particle, enemies, goal_width, FIELD)
    p_z = compute_observation_from_distance_to_goal(distance_to_goal) * \
            compute_observation_from_closest_enemy_distance(distance_to_closest_enemy) * \
            compute_observation_from_angle_to_goal(angle_to_goal, best_shooting_angle) * \
            (1-is_inside_zone(particle, GK_AREA_WITH_MARGINS)) * \
            (1-is_out_of_environment(particle, FIELD))
            #compute_observation_from_free_space_on_goal(best_free_space_on_goal) * \
    
    return p_z

if __name__ == "__main__":
    # config number of enemies and particles
    n_particles = 100
    n_enemies = 6
    delta = robot_diameter/2
    search_factor = 0.01
    minimum_likelihood_difference_to_change_command = 0.01
    best_shooting_angle = get_highest_angle_to_shoot(GK_AREA, 
                                                     goal_width,
                                                     FIELD)
        
    # generate random particles in the environment
    weights, particles = initialize_particles_uniform(n_particles=n_particles, 
                                                      env_limits=PARTICLES_WORKSPACE,
                                                      prohibited_zone_limits=GK_AREA_WITH_MARGINS)
    
    current_command = particles[0]
        
    while True:
        # generate random enemies inside allowed zone
        enemies = generate_enemies(n_enemies=n_enemies, 
                                   env_limits=ENEMIES_WORKSPACE,
                                   gk_area_limits=GK_AREA,
                                   has_goalkeeper=True)
                
        while True:
            # assing weights to the particles
            for idx, particle in enumerate(particles):
                p_z = compute_particle_likelihood(particle, 
                                                  enemies, 
                                                  goal_width, 
                                                  best_shooting_angle, 
                                                  GK_AREA_WITH_MARGINS, 
                                                  FIELD)

                weights[idx] = p_z * weights[idx]
            
            weights = normalize_weights(weights)
            
            weights, particles = systematic_resample(weights, 
                                                     particles, 
                                                     n_particles, 
                                                     delta,
                                                     search_factor,
                                                     PARTICLES_WORKSPACE,
                                                     GK_AREA_WITH_MARGINS)

            average_particle = get_average_particle(weights, particles)

            average_likelihood = compute_particle_likelihood(average_particle,
                                                          enemies, 
                                                          goal_width, 
                                                          best_shooting_angle, 
                                                          GK_AREA_WITH_MARGINS, 
                                                          FIELD)

            current_likelihood = compute_particle_likelihood(current_command,
                                                             enemies, 
                                                             goal_width, 
                                                             best_shooting_angle, 
                                                             GK_AREA_WITH_MARGINS, 
                                                             FIELD)
            
            if average_likelihood - current_likelihood > minimum_likelihood_difference_to_change_command:
                current_command = average_particle
            
            should_break = visualize_with_opencv(current_command,
                                                 current_likelihood,
                                                 average_likelihood,                                                 
                                                 weights, 
                                                 particles,
                                                 enemies,
                                                 FIELD, 
                                                 GK_AREA, 
                                                 PARTICLES_WORKSPACE,
                                                 frame_duration=1)
            if should_break:
                break