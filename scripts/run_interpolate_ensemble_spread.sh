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

num_chunks=8

# Compute chunk size
total=${#variable_lst[@]}
chunk_size=$(( (total + num_chunks - 1) / num_chunks ))

# test
# python ${SCRIPT_DIR}/interpolate_ensemble_spread.py \
#   --variable-lst 100m_u_component_of_wind 100m_v_component_of_wind

# Launch parallel jobs
for ((i = 0; i < total; i += chunk_size)); do
  chunk=("${variable_lst[@]:i:chunk_size}")
  echo "Launching chunk: ${chunk[*]}"
  python ${SCRIPT_DIR}/interpolate_ensemble_spread.py \
    --variable-lst "${chunk[@]}" &
done

wait
