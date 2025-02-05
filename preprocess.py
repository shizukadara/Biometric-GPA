# preprocess_data.py

import pandas as pd
import os
import numpy as np


def count_user_appearances(df):
    """
    Count how many times each user appears in the DataFrame.

    Parameters:
        df (pd.DataFrame): DataFrame containing 'UserId' and 'EmployeeName' columns.

    Returns:
        pd.DataFrame: DataFrame with 'UserId', 'EmployeeName', and their corresponding counts.
    """
    required_columns = ['UserId', 'EmployeeName']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"The DataFrame must contain a '{col}' column.")
    
    user_counts = df['UserId'].value_counts().reset_index()
    user_counts.columns = ['UserId', 'Count']
    user_counts = user_counts.merge(df[['UserId', 'EmployeeName']].drop_duplicates(), on='UserId', how='left')

    print("\nTop Users by Log Count:")
    print(user_counts.nlargest(20, 'Count'))
    
    return user_counts


def calculate_outside_times(merged_df):
    """
    Calculate separate metrics for each user on weekdays and weekends, including total outside times,
    unique OUT days, and total trips.

    Parameters:
        merged_df (pd.DataFrame): Merged DataFrame containing user logs.

    Returns:
        pd.DataFrame: DataFrame with aggregated outside time metrics for each user.
    """
    gate_devices = {'IN': [19, 50], 'OUT': [20, 21]}

    # Filter and classify IN/OUT logs
    gate_df = merged_df[merged_df['DeviceId'].isin(gate_devices['IN'] + gate_devices['OUT'])].copy()
    gate_df['EventType'] = np.where(gate_df['DeviceId'].isin(gate_devices['OUT']), 'OUT', 'IN')
    gate_df = gate_df.sort_values(['UserId', 'LogDate'])

    results = []

    for user_id, user_logs in gate_df.groupby('UserId'):
        metrics = process_user_logs(user_logs)
        metrics['UserId'] = user_id
        metrics['EmployeeName'] = user_logs['EmployeeName'].iloc[0] if not user_logs.empty else ""
        results.append(metrics)
    
    return pd.DataFrame(results)


def process_user_logs(user_logs):
    """
    Process user logs to calculate outside times and trip metrics.

    Parameters:
        user_logs (pd.DataFrame): Logs for a specific user.

    Returns:
        dict: Dictionary of calculated metrics for the user.
    """
    times = {
        'TotalTimeOutsideNonOvernight_Weekday': pd.Timedelta(0),
        'TotalTimeOutsideNonOvernight_Weekend': pd.Timedelta(0),
        'TotalTimeOutsideOvernight_Weekday': pd.Timedelta(0),
        'TotalTimeOutsideOvernight_Weekend': pd.Timedelta(0),
    }

    unique_out_days = {'Weekday': set(), 'Weekend': set()}
    trip_counts = {'Weekday': 0, 'Weekend': 0}
    out_time = None

    for _, row in user_logs.iterrows():
        if row['EventType'] == 'OUT':
            out_time = row['LogDate']
        elif row['EventType'] == 'IN' and out_time is not None:
            in_time = row['LogDate']
            duration = in_time - out_time
            is_weekday = out_time.weekday() < 5
            day_type = 'Weekday' if is_weekday else 'Weekend'

            if out_time.date() != in_time.date():
                times[f'TotalTimeOutsideOvernight_{day_type}'] += duration
            else:
                times[f'TotalTimeOutsideNonOvernight_{day_type}'] += duration
            
            trip_counts[day_type] += 1
            unique_out_days[day_type].add(out_time.date())
            out_time = None

    return {
        **{k: v.total_seconds() / 3600.0 for k, v in times.items()},
        'UniqueOutDays_Weekday': len(unique_out_days['Weekday']),
        'UniqueOutDays_Weekend': len(unique_out_days['Weekend']),
        'TotalTrips_Weekday': trip_counts['Weekday'],
        'TotalTrips_Weekend': trip_counts['Weekend']
    }


def calculate_gym_visits(merged_df):
    """
    Calculate total count of gym visits (DeviceId = 38) for each student.

    Parameters:
        merged_df (pd.DataFrame): Merged DataFrame containing user logs.

    Returns:
        pd.DataFrame: DataFrame with 'UserId' and 'GymnasiumVisitCount' for each student.
    """
    gym_df = merged_df[merged_df['DeviceId'] == 38]
    gym_visit_counts = gym_df['UserId'].value_counts().reset_index()
    gym_visit_counts.columns = ['UserId', 'GymnasiumVisitCount']
    return gym_visit_counts


def load_and_filter_student_data(people_csv_path):
    """Load and filter student data from the people CSV file."""
    try:
        people_df = pd.read_csv(people_csv_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"'{people_csv_path}' not found.")
    
    student_df = people_df[
    ((people_df['Company'] == 'Students') | 
     (people_df['Department'].str.startswith(('UGLE', 'FSP', 'MSC')))) & 
    (people_df['Company'] != 'Withdrawal') & 
    (people_df['EmployeeCode'].astype(str).str.startswith(('21', '22', '23')))
    ]


    return student_df


def load_log_files(raw_data_dir, columns_to_keep):
    """Load all log files from the specified directory."""
    log_files = [f for f in os.listdir(raw_data_dir) if f.lower().endswith(".csv")]
    if not log_files:
        raise FileNotFoundError(f"No log files found in '{raw_data_dir}' directory.")
    
    df_list = []
    for filename in log_files:
        file_path = os.path.join(raw_data_dir, filename)
        try:
            temp_df = pd.read_csv(file_path, usecols=columns_to_keep)
            temp_df['LogDate'] = pd.to_datetime(temp_df['LogDate'].astype(str).str.strip(), errors='coerce', infer_datetime_format=True)
            df_list.append(temp_df)
        except Exception as e:
            print(f"Error loading '{filename}': {e}")
    return pd.concat(df_list, ignore_index=True)

def calculate_library_metrics(merged_df):
    """
    Calculate library visit metrics for each user:
    - Total time spent at the library
    - Total number of library visits
    - Total number of unique visit days

    Parameters:
        merged_df (pd.DataFrame): Merged DataFrame containing user logs.

    Returns:
        pd.DataFrame: DataFrame with 'UserId', 'TotalLibraryTime', 'LibraryVisitCount', and 'UniqueLibraryDays'.
    """
    library_devices_in = [24, 25]  # Library IN devices
    library_device_out = 26       # Library OUT device

    # Filter logs for library devices
    library_df = merged_df[merged_df['DeviceId'].isin(library_devices_in + [library_device_out])].copy()
    library_df['EventType'] = np.where(library_df['DeviceId'] == library_device_out, 'OUT', 'IN')
    library_df = library_df.sort_values(['UserId', 'LogDate'])

    results = []

    for user_id, user_logs in library_df.groupby('UserId'):
        total_time = pd.Timedelta(0)
        visit_count = 0
        unique_days = set()

        in_time = None

        for _, row in user_logs.iterrows():
            if row['EventType'] == 'IN':
                in_time = row['LogDate']
            elif row['EventType'] == 'OUT' and in_time is not None:
                out_time = row['LogDate']
                total_time += out_time - in_time
                visit_count += 1
                unique_days.add(in_time.date())
                in_time = None

        results.append({
            'UserId': user_id,
            'TotalLibraryTime': total_time.total_seconds() / 3600.0,  # Convert to hours
            'LibraryVisitCount': visit_count,
            'UniqueLibraryDays': len(unique_days)
        })

    return pd.DataFrame(results)

def process_cgpa(final_df, cgpa_df, people_df):
    """
    Process the CGPA data by matching UserIds to StudentIds.
    For users in the FSP department, 24xxxx UserIds are used instead of 21xxxx.
    The final CGPA is stored in the 'CGPA' column.

    Parameters:
        final_df (pd.DataFrame): DataFrame containing user metrics.
        cgpa_df (pd.DataFrame): DataFrame containing student CGPA data.
        people_df (pd.DataFrame): DataFrame containing employee data with department info.

    Returns:
        pd.DataFrame: Updated final DataFrame with the 'CGPA' column.
    """

    # Ensure UserId in final_df matches StudentId in cgpa_df
    final_df['UserId'] = final_df['UserId'].astype(str)

    # **Step 1: Identify FSP Department Users**
    fsp_users = people_df[people_df['Department'].str.startswith('FSP', na=False)]['EmployeeCode'].astype(str).tolist()
    fsp_user_ids = [uid if len(uid) == 6 else uid[:2] + '0' + uid[2:] for uid in fsp_users]  # Make sure EmployeeCode is 6-digits (21xxxx -> 210xxx)

    # **Step 2: Handle FSP users - prioritize 24xxxx over 21xxxx**
    final_df['IsFSP'] = final_df['UserId'].isin(fsp_user_ids)
    final_df.loc[final_df['IsFSP'] & final_df['UserId'].str.startswith('21'), 'UserId'] = final_df['UserId'].str.replace('21', '24', 1)

    # **Step 3: Match CGPA for UserIds**
    final_df = final_df.merge(cgpa_df[['StudentId', 'GPA']], left_on='UserId', right_on='StudentId', how='left')

    # Rename the GPA column to "CGPA"
    final_df.rename(columns={'GPA': 'CGPA'}, inplace=True)

    # **Step 4: Handle unmatched UserIds starting with '21xxx'**
    unmatched_ids = final_df[final_df['CGPA'].isna() & final_df['UserId'].str.startswith('21')]
    for idx, row in unmatched_ids.iterrows():
        new_id = row['UserId'].replace('21', '24', 1)  # Replace '21xxx' with '24xxx'
        matching_row = cgpa_df[cgpa_df['StudentId'] == new_id]
        if not matching_row.empty:
            final_df.at[idx, 'UserId'] = new_id
            final_df.at[idx, 'CGPA'] = matching_row['GPA'].values[0]

    # Drop duplicate UserIds
    final_df.drop(columns=['StudentId', 'IsFSP'], inplace=True, errors='ignore')

    return final_df

def calculate_mess_visits(merged_df):
    """
    Calculate the number of breakfast, lunch, and dinner visits for each student in the mess (dining) area.

    Parameters:
        merged_df (pd.DataFrame): Merged DataFrame containing user logs.

    Returns:
        pd.DataFrame: DataFrame with 'UserId', 'BreakfastCount', 'LunchCount', and 'DinnerCount' for each student.
    """
    mess_devices_in = [22, 29]  # Device IDs for mess IN logs
    
    # Filter logs for mess devices
    mess_df = merged_df[merged_df['DeviceId'].isin(mess_devices_in)].copy()
    
    # Sort by UserId and LogDate to ensure chronological order
    mess_df = mess_df.sort_values(['UserId', 'LogDate'])
    
    # Initialize a results list
    results = []
    
    for user_id, user_logs in mess_df.groupby('UserId'):
        # Initialize counters for breakfast, lunch, and dinner visits
        breakfast_count = 0
        lunch_count = 0
        dinner_count = 0
        
        for log_time in user_logs['LogDate']:
            hour = log_time.hour
            
            if 6 <= hour < 10:  # Breakfast time range: 6:00 AM to 10:00 AM
                breakfast_count += 1
            elif 12 <= hour < 15:  # Lunch time range: 12:00 PM to 3:00 PM
                lunch_count += 1
            elif 19 <= hour < 22:  # Dinner time range: 7:00 PM to 10:00 PM
                dinner_count += 1
        
        results.append({
            'UserId': user_id,
            'BreakfastCount': breakfast_count,
            'LunchCount': lunch_count,
            'DinnerCount': dinner_count
        })
    
    return pd.DataFrame(results)



def preprocess_and_save():
    """
    Preprocess raw data, calculate metrics, and merge CGPA data from cgpa.xlsx with the final summary.
    - Modify StudentId in cgpa.xlsx by removing '0' from '2x0xxx'.
    - Update UserId in the main DataFrame (if starting with '21xxx') to check for '24xxx' if 21xxx is not found.
    - Store CGPA in the final DataFrame.
    """
    raw_data_dir = "sem1_raw_data"
    people_csv_path = "people.csv"
    cgpa_file_path = "cgpa.xlsx"
    people_df = pd.read_csv(people_csv_path)

    # Verify raw data directory exists
    if not os.path.isdir(raw_data_dir):
        print(f"Error: '{raw_data_dir}' directory not found.")
        return

    # Load and filter student data
    try:
        student_df = load_and_filter_student_data(people_csv_path)
        student_df.to_csv("filtered_students.csv", index=False)
        print("\nFiltered student data saved to 'filtered_students.csv'.")
    except Exception as e:
        print(f"Error processing student data: {e}")
        return

    # Load logs from raw data directory
    try:
        all_logs = load_log_files(raw_data_dir, columns_to_keep=['DeviceId', 'UserId', 'LogDate'])
        all_logs = all_logs.dropna(subset=['LogDate']).drop_duplicates()
    except Exception as e:
        print(f"Error processing log files: {e}")
        return

    # Merge logs with student data
    try:
        merged_df = all_logs.merge(student_df[['EmployeeCode', 'EmployeeName']],
                                   left_on='UserId', right_on='EmployeeCode', how='inner')
        print(f"\nTotal log entries after merging with student data: {len(merged_df)}")
    except Exception as e:
        print(f"Error merging log data with student data: {e}")
        return

    # Calculate metrics
    try:
        outside_times_df = calculate_outside_times(merged_df)
        gym_visit_counts = calculate_gym_visits(merged_df)
        library_metrics_df = calculate_library_metrics(merged_df)
        mess_metrics_df = calculate_mess_visits(merged_df)
    except Exception as e:
        print(f"Error calculating metrics: {e}")
        return

    # Combine all results into a final DataFrame
    try:
        final_df = outside_times_df.merge(gym_visit_counts, on='UserId', how='left')
        final_df = final_df.merge(library_metrics_df, on='UserId', how='left')
        final_df = final_df.merge(mess_metrics_df, on='UserId', how='left')

        # Fill missing values with defaults
        final_df['GymnasiumVisitCount'] = final_df['GymnasiumVisitCount'].fillna(0).astype(int)
        final_df['TotalLibraryTime'] = final_df['TotalLibraryTime'].fillna(0)
        final_df['LibraryVisitCount'] = final_df['LibraryVisitCount'].fillna(0).astype(int)
        final_df['UniqueLibraryDays'] = final_df['UniqueLibraryDays'].fillna(0).astype(int)
        final_df['BreakfastCount'] = final_df['BreakfastCount'].fillna(0).astype(int)
        final_df['LunchCount'] = final_df['LunchCount'].fillna(0).astype(int)
        final_df['DinnerCount'] = final_df['DinnerCount'].fillna(0).astype(int)

        # Drop users with all calculated metrics as zero
        metrics_columns = final_df.columns.difference(['UserId', 'EmployeeName'])
        non_zero_users = final_df[metrics_columns].sum(axis=1) != 0
        final_df = final_df[non_zero_users]
    except Exception as e:
        print(f"Error combining or filtering metrics: {e}")
        return

    # Add CGPA data from cgpa.xlsx
    try:
        # Load CGPA data and modify StudentId
        cgpa_df = pd.read_excel(cgpa_file_path)
        
        cgpa_df = cgpa_df[~cgpa_df['Program Plan: Program Plan Name'].str.contains('FSP', case=False, na=False)]
        
        # Convert StudentId to string and add '0' at the third position for 5-digit StudentIds
        cgpa_df['StudentId'] = cgpa_df['StudentId'].astype(str).apply(lambda x: x[:2] + '0' + x[2:] if len(x) == 5 else x)
        
        # Convert UserId to string and add '0' at the third position for 5-digit UserIds in final_df
        final_df['UserId'] = final_df['UserId'].astype(str).apply(lambda x: x[:2] + '0' + x[2:] if len(x) == 5 else x)

        # Process CGPA and store in final DataFrame
        final_df = process_cgpa(final_df, cgpa_df, people_df)

        # **Remove people without CGPA**
        original_count = len(final_df)
        final_df = final_df[final_df['CGPA'].notna()]  # Remove rows where CGPA is NaN
        filtered_count = len(final_df)
        print(f"Filtered out {original_count - filtered_count} users without CGPA.")

    except Exception as e:
        print(f"Error processing or merging CGPA data: {e}")
        return



    #FINALIZAING.....DROPPING NAMES....
    # final_df.drop(columns=['EmployeeName', 'UserId'], inplace=True, errors='ignore')
    # Rearrange columns to have EmployeeName, UserId, and CGPA first, and the rest after
    columns_order = ['CGPA'] + [col for col in final_df.columns if col not in ['CGPA']]
    final_df = final_df[columns_order]

    # Rearrange columns to have EmployeeName, UserId, and CGPA first, and the rest after
    columns_order = ['EmployeeName', 'UserId', 'CGPA'] + [col for col in final_df.columns if col not in ['EmployeeName', 'UserId', 'CGPA']]
    final_df = final_df[columns_order]

    # Save the final DataFrame to a CSV file
    output_path = "finalfinal.csv"
    try:
        final_df.to_csv(output_path, index=False)
        print(f"\nFinal summary with CGPA saved to '{output_path}'.")
    except Exception as e:
        print(f"Error saving final summary: {e}")

    print("\nPreprocessing complete.")



if __name__ == "__main__":
    preprocess_and_save()
