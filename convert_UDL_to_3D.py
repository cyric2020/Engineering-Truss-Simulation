import yaml

FILENAME = 'testing_trusses/warren_rise.yaml'

with open(FILENAME, 'r') as f:
    data = yaml.safe_load(f)

forces = data['ExternalForces']

pasteYAML = ''

# Loop through each force
for force in forces:
    force[1] = round(force[1] * 1.6, 2)

    pasteYAML += f'  - [0, 0, {force[1]}]\n'

print(pasteYAML)