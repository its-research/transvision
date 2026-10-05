"""Standalone class semantics; importing this file never loads a model or GT.

The historical numeric detector class 0 and serialized label ``car`` do not
establish a physical class. Vehicle evaluation requires an explicit new bound
producer contract. The native reference excludes exact ``Pedestrian`` inside
project_world_objects, while merging the other entries of params['vehicles'].
"""

VEHICLE_PROTOCOL = 'v2v4real-official-benchmark-vehicle-v1'
NATIVE_VEHICLE_SELECTION = 'params.vehicles_except_exact_obj_type_Pedestrian'


def evaluation_class(dataset, requested='car'):
    if requested == 'car' and dataset in ('spd', 'v2v4real'):
        return requested
    if requested == 'vehicle' and dataset == 'v2v4real':
        return requested
    raise ValueError('unsupported dataset-specific evaluation class')


def native_vehicle(raw_class):
    if not isinstance(raw_class, str) or not raw_class:
        raise ValueError('explicit native obj_type required; no default Car inference')
    return raw_class != 'Pedestrian'


def vehicle_binding():
    """Explicit interpretation of the original single-class detector channel."""
    return dict(evaluation_protocol=VEHICLE_PROTOCOL,
                native_label_source=NATIVE_VEHICLE_SELECTION,
                evaluation_class='vehicle', calibration_evaluation_class='vehicle',
                prediction_class_mapping={'car': 'vehicle'})


def require_vehicle_binding(binding):
    if not isinstance(binding, dict) or any(binding.get(k) != v for k, v in vehicle_binding().items()):
        raise ValueError('explicit native vehicle producer and vehicle calibration binding required')


def require_evaluation_binding(protocol, binding):
    name = evaluation_class(protocol['dataset'], protocol['evaluation_class'])
    if name == 'vehicle':
        require_vehicle_binding(binding)
    return name
