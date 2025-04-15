import folium

def plot_route_on_map(route, coordinates, filename="best_route_map.html"):
    m = folium.Map(location=coordinates['START'], zoom_start=5)
    route_coords = [coordinates['START']] + [coordinates[loc] for loc in route] + [coordinates['START']]

    for loc, (lat, lon) in coordinates.items():
        folium.Marker([lat, lon], popup=loc).add_to(m)

    folium.PolyLine(route_coords, color='blue', weight=3, opacity=0.8).add_to(m)
    m.save(filename)
