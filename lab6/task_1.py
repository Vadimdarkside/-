#1 варіант

# Дані (частотна таблиця для прикладу)
data = {
    "Outlook": {"Overcast": {"Yes": 4, "No": 0}, "Sunny": {"Yes": 3, "No": 2}, "Rain": {"Yes": 3, "No": 2}},
    "Humidity": {"High": {"Yes": 3, "No": 4}, "Normal": {"Yes": 6, "No": 1}},
    "Wind": {"Weak": {"Yes": 3, "No": 3}, "Strong": {"Yes": 6, "No": 2}},
    "Play": {"Yes": 9, "No": 5}
}

# Загальна кількість днів
total_days = sum(data["Play"].values())

# Ймовірності класів (P(Yes) і P(No))
Yes = data["Play"]["Yes"] / total_days
No = data["Play"]["No"] / total_days

#Умовні ймовірності для заданих значень
#Outlook = Overcast
#Overcast Yes
P_Overcast_Yes = data["Outlook"]["Overcast"]["Yes"] / data["Play"]["Yes"]
P_Overcast = (data["Outlook"]["Overcast"]["Yes"]+data["Outlook"]["Overcast"]["No"])/total_days
Overcast_Yes = P_Overcast_Yes * Yes / P_Overcast
#Overcast No
P_Overcast_No = data["Outlook"]["Overcast"]["No"] / data["Play"]["No"]
Overcast_No = P_Overcast_No * No / P_Overcast
print(f"Overcast_Yes: {Overcast_Yes:.3f}")
print(f"Overcast_No: {Overcast_No:.3f}")

#Humidity = High
#Humidity Yes
P_Humidity_Yes = data["Humidity"]["High"]["Yes"] / data["Play"]["Yes"]
P_Humidity = (data["Humidity"]["High"]["Yes"]+data["Humidity"]["High"]["No"])/total_days
Humidity_Yes = P_Humidity_Yes * Yes / P_Humidity
#Humidity No
P_Humidity_No = data["Humidity"]["High"]["No"] / data["Play"]["No"]
Humidity_No = P_Humidity_No * No / P_Humidity
print(f"Humidity_Yes: {Humidity_Yes:.3f}")
print(f"Humidity_No: {Humidity_No:.3f}")


#Wind = Weak
#Wind Yes
P_Wind_Yes = data["Wind"]["Weak"]["Yes"] / data["Play"]["Yes"]
P_Wind = (data["Wind"]["Weak"]["Yes"]+data["Wind"]["Weak"]["No"])/total_days
Wind_Yes = P_Wind_Yes * Yes / P_Wind
#Wind No
P_Wind_No = data["Wind"]["Weak"]["No"] / data["Play"]["No"]
Wind_No = P_Wind_No * No / P_Wind
print(f"Humidity_Yes: {Wind_Yes:.3f}")
print(f"Humidity_No: {Wind_No:.3f}")

# Розрахунок P(Yes) і P(No) для заданих умов
Yes_given_data = Overcast_Yes * Humidity_Yes * Wind_Yes * Yes
No_given_data = Overcast_No * Humidity_No * Wind_No * No

# Нормалізація ймовірностей
Yes_normalized = Yes_given_data / (Yes_given_data + No_given_data)
No_normalized = No_given_data / (Yes_given_data + No_given_data)

# Виведення результатів
print(f"Ймовірність, що матч відбудеться (P(Yes)): {Yes_normalized:.2f}")
print(f"Ймовірність, що матч не відбудеться (P(No)): {No_normalized:.2f}")
