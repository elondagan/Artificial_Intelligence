# build a taxi fleet management system using forward search and heuristics  (2023)

**Objection:**

Take the role of the owner of a taxi business and manage the delivery of passengers to their destinations in the shortest time possible

**Environment:**

The environment is a rectangular grid - given as a list of lists. Each point on a grid represents an area. An area can be either passable or impassable for the taxis. On the grid, there are passengers waiting to be delivered to their destinations. Moreover, there are gas stations, where the taxis can get additional fuel to continue serving the customers.

**Actions:**

1.	Move on the grid up to one tile vertically or horizontally (taxis cannot move diagonally). The taxis cannot move to an impassable tile. Every movement takes 1 unit of fuel, and cannot be performed if the fuel level of a certain taxi is equal to 0. This is the only action that decreases the amount of fuel, and the only action that cannot be performed when the taxi has no fuel.
2.	Pick up passengers if they are on the same tile as the taxi. The number of passengers in the taxi at any given turn cannot exceed this taxi’s capacity. 
3.	Drop off passengers on the same tile as the taxi. The passenger can only be dropped off on his destination tile and will refuse to leave the vehicle otherwise. 
4.	Refuel the taxi. Refueling can be performed only at gas stations and brings the amount of fuel back to the maximum capacity. 
5.	Wait. Does not change anything about the taxi.
All 5 actions above are atomic actions. Taxis can act simultaneously.
