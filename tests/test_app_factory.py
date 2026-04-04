from api.app_factory import create_app


def test_create_app_registers_core_routes():
    app = create_app()
    paths = {route.path for route in app.router.routes}
    assert "/health" in paths
    assert "/auth/login" in paths
    assert "/predict" in paths
    assert "/history/{patient_id}" in paths
    assert "/report/generate" in paths
