from Box2D.b2 import contactListener


class FrictionDetector(contactListener):
    def __init__(self, env, lap_complete_percent, road_color):
        contactListener.__init__(self)
        self.env = env
        self.lap_complete_percent = lap_complete_percent
        self.road_color = road_color

    def BeginContact(self, contact):
        self._contact(contact, True)

    def EndContact(self, contact):
        self._contact(contact, False)

    def _extract(self, contact):
        u1 = contact.fixtureA.body.userData
        u2 = contact.fixtureB.body.userData

        tile, obj = None, None

        if u1 and "road_friction" in u1.__dict__:
            tile = u1
            obj = u2

        if u2 and "road_friction" in u2.__dict__:
            tile = u2
            obj = u1

        return tile, obj

    def _update(self, tile, car_id):
        tile.road_visited[car_id] = True
        self.env.reward[car_id] += 1000.0 / len(self.env.track)
        self.env.tile_visited_count[car_id] += 1

        # Lap is considered completed if enough % of the track was covered
        if (
            tile.idx == 0
            and self.env.tile_visited_count[car_id] / len(self.env.track)
            > self.lap_complete_percent
        ):
            self.env.new_lap = True

    def _contact(self, contact, begin):
        tile, obj = self._extract(contact)
        if not tile:
            return

        tile.color = self.road_color

        # This check seems to implicitly make sure that we only look at wheels as the tiles
        # attribute is only set for wheels in car_dynamics.py.
        if not obj or "tiles" not in obj.__dict__:
            return

        if begin:
            obj.tiles.add(tile)
            # print tile.road_friction, "ADD", len(obj.tiles)
            if not tile.road_visited[obj.car_id]:
                self._update(tile, obj.car_id)
        else:
            obj.tiles.remove(tile)
            # print tile.road_friction, "DEL", len(obj.tiles) -- should delete to zero when on grass (this works)
