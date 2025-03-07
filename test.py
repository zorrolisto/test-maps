import pandas as pd

# Load the Excel file
file_path = "389657_21311_1740606018 (1).xlsx"  # Change to your file path
df = pd.read_excel(file_path, engine="openpyxl")

# Show all keys
#print(df.keys())

# Apply multiple filters correctly
filtered_df = df[
    (df["FECHA AO"] == "21/02/2025") &
    (df["FECHA VISITA 1"] == "22/02/2025") &
    (df["DESTINO"] == "LIM") &
    (df["LOCALIDAD"] == "MIRAFLORES - LIMA - LIMA")
    #(df["LOCALIDAD"] == "MIRAFLORES - LIMA - LIMA") &
    #(df["FECHA AO"] == "21/02/2025") &
    #(df["FECHA VISITA 1"] == "22/02/2025") &
    #(df["DESTINO"] == "LIM")
]

#find "GUIA" = "WYB317190891"
print("ENCONtre la guia")
print(filtered_df[filtered_df["GUIA"] == "WYB317190891"])

# Save the filtered data to a new Excel file
# filtered_df.to_excel("filtered_data.xlsx", index=False)

# Show first 5 rows
#print(filtered_df.head())

# Show number of all rows
#print(filtered_df.shape[0])
