# env dimensions
field_length = 12000
field_width = 9000
goalkeeper_area_length = 1800
goalkeeper_area_width = 3600
boundary_width = 300
robot_diameter = 200
goal_width = 1800

# field limits
field_xmin = -field_length/2
field_xmax = field_length/2
field_ymin = -field_width/2
field_ymax = field_width/2

# gk area limits
gk_area_xmin = field_length/2-goalkeeper_area_length
gk_area_xmax = field_length/2
gk_area_ymin = -goalkeeper_area_width/2
gk_area_ymax = goalkeeper_area_width/2
    
# particles workspace limits
particles_workspace_xmin = 0
particles_workspace_xmax = field_length/2 - 500
particles_workspace_ymin = -(field_width/2 - 0)
particles_workspace_ymax = field_width/2 - 0

# enemies workspace limits
enemies_workspace_xmin = gk_area_xmin - 500
enemies_workspace_xmax = gk_area_xmax - 1000
enemies_workspace_ymin = gk_area_ymin - 500
enemies_workspace_ymax = gk_area_ymax + 500

# define envs
FIELD = field_xmin, field_xmax, field_ymin, field_ymax
GK_AREA = gk_area_xmin, gk_area_xmax, gk_area_ymin, gk_area_ymax
PARTICLES_WORKSPACE = particles_workspace_xmin, particles_workspace_xmax, particles_workspace_ymin, particles_workspace_ymax
ENEMIES_WORKSPACE = enemies_workspace_xmin, enemies_workspace_xmax, enemies_workspace_ymin, enemies_workspace_ymax

# define lowest value
EPS = 10e-6