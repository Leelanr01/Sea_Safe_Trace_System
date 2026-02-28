import streamlit as st 
import cv2
import pandas as pd 
import folium
from geopy.distance import geodesic
import matplotlib.pyplot as plt
from streamlit.components.v1 import html

image_path = "img_0007.jpg"
image = cv2.imread(image_path)

if image is None:
    st.write("Image file not found")
    exit()
st.write("Image Loaded Successfully")

gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

_, oil_mask = cv2.threshold(gray_image, 60, 255, cv2.THRESH_BINARY_INV)

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
oil_mask = cv2.morphologyEx(oil_mask, cv2.MORPH_OPEN, kernel)
oil_mask = cv2.morphologyEx(oil_mask, cv2.MORPH_CLOSE, kernel)

contours, _ = cv2.findContours(oil_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

oil_area = 0
for cnt in contours:
    oil_area += cv2.contourArea(cnt)
st.write(f"Estimated Oil Spill Area: {oil_area}")

plt.imshow(oil_mask, cmap="gray")
plt.title("Detected Oil Spill Region")
plt.axis("off")
st.pyplot()

st.write("Oil detection Completed")

csv_path = "AIS_2022_03_31.csv"
ais_data = pd.read_csv(csv_path)

st.write("\nAvailable columns: ")
st.write(ais_data.columns)

ais_data = ais_data.dropna(subset=["LAT", "LON"])
st.write("\n Total vessels loaded:", len(ais_data))

oil_location = (29.78, -95.08)
st.write("\nChecking vessels near oil spill...\n") 

suspicious_vessels = []

for i, row in ais_data.iterrows():
    vessel_location = (row["LAT"], row["LON"] )
    distance_km = geodesic(oil_location, vessel_location).km
    
    if distance_km < 50:
        speed = row["SOG"]
        vessel_type = row.get("VesselType", "Unknown")
        distance_score = 50 - distance_km
        speed_score = max(0, 20 - speed)
    else:
        type_score = 0
        if vessel_type == "Tanker":
            type_score = 30
        elif vessel_type == "Cargo":
            type_score = 15
        else:
            type_score = 0
        total_score = distance_score + speed_score + type_score
        
        suspicious_vessels.append({"name": row["VesselName"],
                                   "distance": distance_km,
                                   "speed": speed,
                                   "type": vessel_type,
                                   "risk": total_score,
                                   "lat": row["LAT"],
                                   "lon": row["LON"]})
suspicious_vessels = sorted(suspicious_vessels, key=lambda x: x["risk"], reverse=True)

if suspicious_vessels:
    main_suspect = suspicious_vessels[0]
    st.write("\n MOST PROBABLE RESPONSIBLE VESSEL:")
    st.write(main_suspect)
else:
    st.write("No vessels found nearby.")
         
map_object = folium.Map(location=oil_location, zoom_start=6)

folium.Marker(
    oil_location,
    popup="Oil Spill Area",
    icon=folium.Icon(color="red")
).add_to(map_object)

folium.Circle(
    location=oil_location,
    radius=50000,
    color='red',
    fill=True,
    fill_opacity=0.1).add_to(map_object)

for vessel in suspicious_vessels:
    color = "blue"
    if suspicious_vessels and vessel == suspicious_vessels[0]:
        color = "black"
    folium.Marker(
        [vessel["lat"], vessel["lon"]],
        popup=f'{vessel["name"]} | Risk Score:{vessel["risk"]}',
        icon=folium.Icon(color=color)).add_to(map_object)

map_object.save("oil_spill_map.html")
st.write("Map saved")
    
