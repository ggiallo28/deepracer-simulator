from shapely.geometry import Polygon, LineString, Point
import math, statistics
import numpy as np

GEOM_AREA_THRESHOLD = 5


class DeepRacerParameters:
    def __init__(self, env):
        self.deep_racer_env = env

    def _get_wayline(self):
        return LineString(
            [
                Point([c[0] for c in geom.centroid.xy])
                for geom in self.deep_racer_env.road_poly_shapely
                if geom.area > GEOM_AREA_THRESHOLD
            ]
        )

    def _get_polygon_axis(self, geom):
        mbr_points = list(zip(*geom.minimum_rotated_rectangle.exterior.coords.xy))
        mbr_lengths = [
            LineString((mbr_points[i], mbr_points[i + 1])).length
            for i in range(len(mbr_points) - 1)
        ]
        minor_axis = min(mbr_lengths)
        major_axis = max(mbr_lengths)

        return minor_axis, major_axis

    def _get_track_width(self):
        axis_length = [
            self._get_polygon_axis(geom)[1]
            for geom in self.deep_racer_env.road_poly_shapely
            if geom.area > GEOM_AREA_THRESHOLD
        ]
        return statistics.mean(axis_length)

    def _get_car_position(self, car):
        car_pos = np.array(car.hull.position).reshape((1, 2))
        car_x_pos = float(car_pos[:, 0])
        car_y_pos = float(car_pos[:, 1])
        return car_x_pos, car_y_pos

    def _get_car_velocity(self, car):
        linear_velocity = car.hull.linearVelocity
        return np.linalg.norm(linear_velocity), linear_velocity

    def _get_car_angle(self, car):
        velocity, linear_velocity = self._get_car_velocity(car)
        if velocity > 0.5:
            car_angle = -math.atan2(linear_velocity[0], linear_velocity[1])
        else:
            car_angle = car.hull.angle
        car_angle = (car_angle + (2 * np.pi)) % (2 * np.pi)
        return math.degrees(car_angle) - 180

    def _get_closest_waypoints(self, car_x_pos, car_y_pos, params):
        car_pos_as_point = Point(car_x_pos, car_y_pos)
        distances = [
            {
                "distance": Point(waypoint).distance(car_pos_as_point),
                "waypoint_index": index,
            }
            for index, waypoint in enumerate(zip(*params["_wayline"].xy))
        ]
        return sorted(distances, key=lambda x: x["distance"])[:2]

    def _get_distance_from_center(self, car_x_pos, car_y_pos, params):
        car_pos_as_point = Point(car_x_pos, car_y_pos)
        distance_from_center = params["_wayline"].distance(car_pos_as_point)
        return distance_from_center

    def _get_track_length(self, params):
        return params["_wayline"].length

    def _get_all_wheels_on_track(self, car):
        on_grass = []
        for wheel in car.wheels:
            wheel_point = Point(wheel.position)
            on_grass += [
                not np.array(
                    [
                        wheel_point.within(polygon)
                        for polygon in self.deep_racer_env.road_poly_shapely
                    ]
                ).any()
            ]
        return not np.array(on_grass).any()

    def _get_steering_angle(self, car, params):
        wheel_angles = []
        for wheel in car.wheels[:2]:
            wheel_angles += [wheel.joints[0].joint.angle]
        wheel_angle = max(wheel_angles)
        return math.degrees(wheel_angle)

    def _get_left_line(self):
        return LineString(
            [
                Point([val[0] for val in geom.exterior.xy])
                for geom in self.deep_racer_env.road_poly_shapely
                if geom.area > GEOM_AREA_THRESHOLD
            ]
        )

    def _get_right_line(self):
        return LineString(
            [
                Point([val[2] for val in geom.exterior.xy])
                for geom in self.deep_racer_env.road_poly_shapely
                if geom.area > GEOM_AREA_THRESHOLD
            ]
        )

    def _get_is_left_of_center(self, car_x_pos, car_y_pos, params):
        car_pos = Point(car_x_pos, car_y_pos)
        return params["_left_line"].distance(car_pos) < params["_right_line"].distance(
            car_pos
        )

    def _get_progress(self, car_id):
        car_tile_visited_count = self.deep_racer_env.tile_visited_count[car_id]
        return car_tile_visited_count / len(self.deep_racer_env.track)

    def _get_is_reversed(self, car_id):
        return self.deep_racer_env.driving_backward[car_id]

    def get(self):
        params = {}
        params["steps"] = self.deep_racer_env.steps
        params["_wayline"] = self._get_wayline()
        # params['waypoints'] = list(zip(*params['_wayline'].xy))
        params["time"] = self.deep_racer_env.t
        params["track_width"] = self._get_track_width()
        params["track_length"] = self._get_track_length(params)
        params["_left_line"] = self._get_right_line()
        params["_right_line"] = self._get_left_line()
        params["_track"] = Polygon(
            [*params["_left_line"].coords, *params["_right_line"].coords]
        )
        params["cars"] = {}

        for car_id, car in enumerate(self.deep_racer_env.cars):
            car_params = {}

            car_x_pos, car_y_pos = self._get_car_position(car)
            car_params.setdefault("x", car_x_pos)
            car_params.setdefault("y", car_y_pos)
            car_params.setdefault("_xy", Point(car_x_pos, car_y_pos))

            velocity, _ = self._get_car_velocity(car)
            car_params.setdefault("speed", velocity)

            car_angle = self._get_car_angle(car)
            car_params.setdefault("heading", car_angle)

            closest_waypoints = self._get_closest_waypoints(
                car_x_pos, car_y_pos, params
            )
            car_params.setdefault("closest_waypoints", closest_waypoints)
            # TODO: Check if needs to reverse
            # The zero-based indices of the two neighboring waypoints closest to the agent's current position of (x, y).
            # The distance is measured by the Euclidean distance from the center of the agent.
            # The first element refers to the closest waypoint behind the agent and the second element refers the closest waypoint in front of the agent.
            # Max is the length of the waypoints list. In the illustration shown in waypoints, the closest_waypoints would be [16, 17].

            distance_from_center = self._get_distance_from_center(
                car_x_pos, car_y_pos, params
            )
            car_params.setdefault("distance_from_center", distance_from_center)

            all_wheels_on_track = self._get_all_wheels_on_track(car)
            car_params.setdefault("all_wheels_on_track", all_wheels_on_track)

            steering_angle = self._get_steering_angle(car, params)
            car_params.setdefault("steering_angle", steering_angle)

            is_offtrack = self.deep_racer_env.driving_on_grass[car_id]
            car_params.setdefault("is_offtrack", is_offtrack)

            is_left_of_center = self._get_is_left_of_center(
                car_x_pos, car_y_pos, params
            )
            car_params.setdefault("is_left_of_center", is_left_of_center)
            # TODO: Check if needs to reverse

            progress = self._get_progress(car_id)
            car_params.setdefault("progress", progress)

            is_reversed = self._get_is_reversed(car_id)
            car_params.setdefault("is_reversed", is_reversed)

            params["cars"].setdefault(car_id, car_params)

        return params


if __name__ == "__main__":
    from multi_deepracer import MultiDeepRacer
    import json

    NUM_CARS = 2

    env = MultiDeepRacer(
        num_agents=NUM_CARS,
        direction="CCW",
        use_random_direction=True,
        backwards_flag=True,
        h_ratio=0.25,
        use_ego_color=True,
    )
    env.reset()

    params = DeepRacerParameters(env).get()
    env.close()
