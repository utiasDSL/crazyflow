import numpy as np
import pytest
from conftest import skip_if_headless

from crazyflow import Sim
from crazyflow.sim.visualize import draw_capsule, draw_line, draw_points


@pytest.mark.unit
@skip_if_headless
def test_draw_capsule(device: str):
    """Test drawing a capsule and verify it changes the rendering."""
    sim = Sim(device=device)
    # Render without drawing
    sim.render(mode="rgb_array", width=120, height=90)  # Warm up the renderer
    img_before = sim.render(mode="rgb_array", width=120, height=90)
    # Draw a capsule and render
    p1 = np.array([0.0, 0.0, 0.5])
    p2 = np.array([0.5, 0.5, 1.0])
    rgba = np.array([1.0, 0.0, 0.0, 1.0])
    draw_capsule(sim, p1, p2, radius=0.05, rgba=rgba)
    img_after = sim.render(mode="rgb_array", width=120, height=90)
    # Verify that the image changed
    assert not np.array_equal(img_before, img_after), "Drawing capsule should change the rendering"
    sim.close()


@pytest.mark.unit
@skip_if_headless
def test_draw_capsule_cylinder(device: str):
    """Test drawing a cylinder."""
    sim = Sim(device=device)
    sim.render(mode="rgb_array", width=120, height=90)  # Warm up the renderer
    img_before = sim.render(mode="rgb_array", width=120, height=90)
    # Draw a cylinder instead of capsule
    p1 = np.array([0.0, 0.0, 0.5])
    p2 = np.array([0.0, 0.0, 1.0])
    rgba = np.array([0.0, 1.0, 0.0, 1.0])
    draw_capsule(sim, p1, p2, radius=0.05, rgba=rgba, cylinder=True)
    img_after = sim.render(mode="rgb_array", width=120, height=90)
    assert not np.array_equal(img_before, img_after), "Drawing cylinder should change the rendering"
    sim.close()


@pytest.mark.unit
@skip_if_headless
def test_draw_line(device: str):
    """Test drawing a line and verify it changes the rendering."""
    sim = Sim(device=device)
    sim.render(mode="rgb_array", width=120, height=90)  # Warm up the renderer
    img_before = sim.render(mode="rgb_array", width=120, height=90)
    # Draw a line with multiple points
    points = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 2.0]])
    rgba = np.array([0.0, 0.0, 1.0, 1.0])
    draw_line(sim, points, rgba=rgba, start_size=5.0, end_size=2.0)
    img_after = sim.render(mode="rgb_array", width=120, height=90)
    assert not np.array_equal(img_before, img_after), "Drawing line should change the rendering"
    sim.close()


@pytest.mark.unit
@skip_if_headless
def test_draw_points(device: str):
    """Test drawing points and verify it changes the rendering."""
    sim = Sim(device=device)
    sim.render(mode="rgb_array", width=120, height=90)  # Warm up the renderer
    img_before = sim.render(mode="rgb_array", width=120, height=90)
    # Draw multiple points
    points = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.5], [0.0, 0.0, 1.0]])
    rgba = np.array([1.0, 1.0, 0.0, 1.0])
    draw_points(sim, points, rgba=rgba, size=0.02)
    img_after = sim.render(mode="rgb_array", width=120, height=90)
    assert not np.array_equal(img_before, img_after), "Drawing points should change the rendering"
    sim.close()


@pytest.mark.unit
@skip_if_headless
def test_draw_combined(device: str):
    """Test drawing multiple visualization elements together."""
    sim = Sim(device=device)
    p1 = np.array([0.0, 0.0, 0.5])
    p2 = np.array([0.2, 0.0, 0.8])
    draw_capsule(sim, p1, p2, radius=0.03, rgba=np.array([1.0, 0.0, 0.0, 1.0]))
    line_points = np.array([[0.3, 0.0, 0.5], [0.3, 0.2, 0.7], [0.5, 0.2, 0.9]])
    draw_line(sim, line_points, rgba=np.array([0.0, 1.0, 0.0, 1.0]))
    points = np.array([[0.6, 0.0, 0.6], [0.7, 0.1, 0.7], [0.8, 0.0, 0.8]])
    draw_points(sim, points, rgba=np.array([0.0, 0.0, 1.0, 1.0]), size=0.025)
    sim.render(mode="rgb_array", width=120, height=90)
    sim.close()
