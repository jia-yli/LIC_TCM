SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

variable_lst=(
  # "100m_u_component_of_wind"
  # "100m_v_component_of_wind"
  "10m_u_component_of_wind"
  # "10m_v_component_of_wind"
  # "2m_dewpoint_temperature"
  "2m_temperature"
  # "ice_temperature_layer_1"
  # "ice_temperature_layer_2"
  "ice_temperature_layer_3"
  # "ice_temperature_layer_4"
  # "maximum_2m_temperature_since_previous_post_processing"
  "mean_sea_level_pressure"
  # "minimum_2m_temperature_since_previous_post_processing"
  # "sea_surface_temperature"
  "skin_temperature"
  # "surface_pressure"
  # "total_precipitation"
)

for variable in "${variable_lst[@]}"; do
  sbatch --job-name=lic-tcm-train-${variable} \
    --export=ALL,VAR=${variable} \
    --dependency=singleton \
    ${SCRIPT_DIR}/launch_train_finetune.sbatch
done
