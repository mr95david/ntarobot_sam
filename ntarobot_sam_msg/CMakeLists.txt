cmake_minimum_required(VERSION 3.8)
project(ntarobot_sam_msg)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Nota: ESTE PAQUETE CREA LOS VALORES DE ENTRADA PARA UN SERVICIO DE EJECUCION DEL PAQUETE DE SAM
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Seccion de librerias requeridas para el funcionamiento del paquete
find_package(ament_cmake REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
# Paquete para generacion de valores por defecto
find_package(rosidl_default_generators REQUIRED)

# Agregar servicio de ros2 
rosidl_generate_interfaces(${PROJECT_NAME}
  "srv/Segmentacion.srv"
  "srv/SegmentAuto.srv"
  DEPENDENCIES geometry_msgs std_msgs sensor_msgs
)


if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
