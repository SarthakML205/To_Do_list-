#author- sarthak .
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
import random
import streamlit as st

# Initialize an empty task list
tasks = pd.DataFrame(columns=['description', 'priority'])

# Load pre-existing tasks from a CSV file (if any)
try:
    tasks = pd.read_csv('tasks.csv')
except FileNotFoundError:
    pass

# Handle missing values in task descriptions
tasks['description'] = tasks['description'].fillna('')

# Function to save tasks to a CSV file
def save_tasks():
    tasks.to_csv('tasks.csv', index=False)

# Train the task priority classifier
vectorizer = CountVectorizer()
clf = MultinomialNB()
model = make_pipeline(vectorizer, clf)
model.fit(tasks['description'], tasks['priority'])

# Function to add a task to the list
def add_task(description, priority):
    global tasks  # Declare tasks as a global variable
    new_task = pd.DataFrame({'description': [description], 'priority': [priority]})
    tasks = pd.concat([tasks, new_task], ignore_index=True)
    save_tasks()

# Function to remove a task by description
def remove_task(description):
    global tasks  # Declare tasks as a global variable
    tasks = tasks[tasks['description'] != description]
    save_tasks()

# Function to list all tasks
def list_tasks():
    if tasks.empty:
        st.write("No tasks available.")
    else:
        st.write(tasks)

# Function to recommend a task based on machine learning
def recommend_task():
    global tasks, model  # Declare tasks and model as global variables
    if not tasks.empty:
        # Get high-priority tasks
        high_priority_tasks = tasks[tasks['priority'] == 'High']
        
        if not high_priority_tasks.empty:
            # Choose a random high-priority task using machine learning model
            random_task_index = random.choice(high_priority_tasks.index)
            random_task_description = high_priority_tasks.loc[random_task_index, 'description']
            st.write(f"Recommended task: {random_task_description} - Priority: High")
        else:
            st.write("No high-priority tasks available for recommendation.")
    else:
        st.write("No tasks available for recommendations.")

# Streamlit app
st.title("Task Management App")

# Main menu
choice = st.sidebar.selectbox("Select an option:", ["Add Task", "Remove Task", "List Tasks", "Recommend Task", "Exit"])

if choice == "Add Task":
    st.header("Add Task")
    description = st.text_input("Enter task description:")
    priority = st.selectbox("Enter task priority:", ["Low", "Medium", "High"])
    
    if st.button("Add Task"):
        add_task(description, priority)
        st.success("Task added successfully.")

elif choice == "Remove Task":
    st.header("Remove Task")
    description = st.text_input("Enter task description to remove:")
    
    if st.button("Remove Task"):
        remove_task(description)
        st.success("Task removed successfully.")

elif choice == "List Tasks":
    st.header("List Tasks")
    list_tasks()

elif choice == "Recommend Task":
    st.header("Recommend Task")
    recommend_task()

elif choice == "Exit":
    st.write("Goodbye!")
