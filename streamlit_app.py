import streamlit as st
import pandas as pd
import statsmodels.api as sm
import numpy as np
import itertools
import warnings
import os
import time
from concurrent.futures import ProcessPoolExecutor

# Suppressing FutureWarnings regarding pandas deprecations
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

# Predefined year selections for each window
predefined_years = {
    "2001-2023": list(range(2001, 2024)),
    "2005 - 2023 excluding 2009, 2016": [year for year in range(2005, 2024) if year not in {2009, 2016}],
    "2001-2023 excluding 2020, 2021, 2022": [year for year in range(2001, 2024) if year not in {2020, 2021, 2022}],
    "2001-2019": list(range(2001, 2020)),
    "2001-2023 excluding 2009, 2013, 2020, 2021, 2022": [year for year in range(2001, 2024) if year not in {2009, 2013, 2020, 2021, 2022}]
}

def run_regression(df, y_col, selected_x_vars):
    try:
        Y = df[y_col].astype(float)
        X = df[selected_x_vars].astype(float)
        X = sm.add_constant(X)
        model = sm.OLS(Y, X).fit()
        return model
    except KeyError as e:
        st.error(f"KeyError in run_regression: {e}")
        raise
    except Exception as e:
        st.error(f"Unexpected error in run_regression: {e}")
        raise

def format_regression_output(model):
    try:
        summary_df = pd.read_html(model.summary().tables[1].as_html(), header=0, index_col=0)[0]
        return summary_df
    except Exception as e:
        st.error(f"Error in format_regression_output: {e}")
        raise

def calculate_anova_table(model):
    try:
        sse = model.ssr  # Sum of squared residuals
        ssr = model.ess  # Explained sum of squares
        sst = ssr + sse  # Total sum of squares
        dfe = model.df_resid  # Degrees of freedom for error
        dfr = model.df_model  # Degrees of freedom for regression
        dft = dfr + dfe  # Total degrees of freedom

        mse = sse / dfe  # Mean squared error
        msr = ssr / dfr  # Mean squared regression

        f_stat = msr / mse  # F-statistic
        p_value = model.f_pvalue  # P-value for the F-statistic

        anova_table = pd.DataFrame({
            'df': [dfr, dfe, dft],
            'SS': [ssr, sse, sst],
            'MS': [msr, mse, np.nan],
            'F': [f_stat, np.nan, np.nan],
            'Significance F': [p_value, np.nan, np.nan]
        }, index=['Regression', 'Residual', 'Total'])

        return anova_table
    except Exception as e:
        st.error(f"Error in calculate_anova_table: {e}")
        raise

def process_scenario(scenario_name, years, df, y_col):
    try:
        df_selected = df[df['Year'].isin(years)]
        variables = df.columns[2:].tolist()  # Assuming variables start from column C onwards
        num_variables = len(variables)

        combinations = itertools.chain.from_iterable(
            itertools.combinations(variables, r) for r in range(1, num_variables + 1)
        )

        scenario_results = []

        for idx, selected_x_vars in enumerate(combinations, start=1):
            columns_to_keep = ['Year', y_col] + list(selected_x_vars)
            df_selected_sub = df_selected[columns_to_keep]
            model = run_regression(df_selected_sub, y_col, selected_x_vars)
            if model:
                output_df = format_regression_output(model)
                anova_table = calculate_anova_table(model)
                scenario_results.append((output_df, years, y_col, model, anova_table, selected_x_vars, idx))

        return scenario_name, scenario_results
    except Exception as e:
        st.error(f"Error in process_scenario for {scenario_name}: {e}")
        raise

class RegressionApp:
    def __init__(self):
        self.df = None
        self.variables = []
        self.scenarios = dict(predefined_years)
        self.num_combinations = 0

    def choose_file(self):
        file = st.file_uploader("Upload Excel file", type=["xlsx"])
        if file:
            self.df = pd.read_excel(file, sheet_name="Sheet1", dtype=float)
            self.variables = self.df.columns[2:].tolist()  # Assuming variables start from column C onwards
            st.write("### Columns in the uploaded file:")
            st.write(self.df.columns.tolist())

    def show_variable_selection(self):
        if self.df is None:
            st.warning("Please upload an Excel file first.")
            return

        self.variables = self.df.columns[2:].tolist()  # Assuming variables start from column C onwards
        self.num_combinations = sum([len(list(itertools.combinations(self.variables, i))) for i in range(1, len(self.variables) + 1)])

        st.subheader(f"{len(self.variables)} variables found.")
        st.subheader(f"Regression will create {self.num_combinations} variable combinations.")

        st.write("### Variables:")
        for var in self.variables:
            st.write(f"- {var}")

    def display_scenarios(self):
        all_years = list(range(2001, 2024))
        scenario_df = pd.DataFrame(columns=all_years)

        for name, years in self.scenarios.items():
            scenario_df.loc[name] = ['•' if year in years else '' for year in all_years]

        try:
            styled_scenario_df = scenario_df.T.style.set_properties(**{'text-align': 'center'}).set_table_styles([dict(selector='th', props=[('text-align', 'center')])])
            st.dataframe(styled_scenario_df)
        except Exception as e:
            st.error(f"Error displaying styled DataFrame: {e}")

    def run_regression_scenarios(self):
        if self.df is None:
            st.warning("Please upload an Excel file first.")
            return

        # Check if 'Year' column exists
        if 'Year' not in self.df.columns:
            st.error("The 'Year' column is missing from the uploaded file.")
            return

        all_results = []

        # Estimate time and set up countdown
        total_combinations = sum([len(list(itertools.combinations(self.df.columns[2:], i))) for i in range(1, len(self.df.columns[2:]) + 1)])
        estimated_time = total_combinations * 0.05  # Assume each combination takes 0.05 seconds
        start_time = time.time()

        st.write(f"Estimated processing time: {estimated_time:.2f} seconds")
        progress_bar = st.progress(0)
        progress_text = st.empty()
        lines_processed_text = st.empty()

        def update_progress(lines_processed, total_lines):
            elapsed_time = time.time() - start_time
            progress = min((lines_processed / total_lines) * 100, 100)
            time_left = max(estimated_time - elapsed_time, 0)
            progress_bar.progress(progress / 100)
            progress_text.write(f"Processing: {int(progress)}% complete")
            lines_processed_text.write(f"Lines processed: {lines_processed} out of {total_lines} ({time_left:.2f} seconds left)")

        with ProcessPoolExecutor() as executor:
            futures = []
            total_lines = 0

            for scenario_name, years in self.scenarios.items():
                if not years:  # If years selection is empty, use predefined years
                    years = predefined_years[scenario_name]
                total_lines += len(list(itertools.chain.from_iterable(itertools.combinations(self.df.columns[2:], r) for r in range(1, len(self.df.columns[2:]) + 1))))
                futures.append(executor.submit(process_scenario, scenario_name, years, self.df, self.df.columns[1]))

            lines_processed = 0
            for future in futures:
                try:
                    result = future.result()
                    all_results.append(result)
                    lines_processed += len(result[1])
                    update_progress(lines_processed, total_lines)
                except Exception as e:
                    st.error(f"Error processing future result: {e}")
                    st.error(e)  # Persistently show the error

        st.success("Regression analysis completed!")
        self.show_combined_results_window(all_results)

    def show_combined_results_window(self, all_results):
        st.session_state["results"] = all_results
        st.experimental_rerun()

    def display_results_page(self):
        if "results" not in st.session_state:
            st.write("No results to display. Please run the regression scenarios first.")
            return

        all_results = st.session_state["results"]

        # Prepare to display scenarios
        tab_titles = list(predefined_years.keys())
        tabs = st.tabs(tab_titles)

        for tab, (scenario_name, scenario_results) in zip(tabs, all_results):
            with tab:
                st.subheader(f"Scenario: {scenario_name}")

                # Display top 20 lines preview thumbnail
                if scenario_results:
                    preview_data = []
                    for result in scenario_results[:20]:  # Preview only top 20 results
                        output_df, selected_years, y_variable_name, model, anova_table, selected_x_vars, idx = result

                        preview_data.append({
                            "Selected Years": ', '.join(map(str, selected_years)),
                            "R Square": f"{model.rsquared:.4f}",
                            "Adjusted R Square": f"{model.rsquared_adj:.4f}",
                            "Observations": f"{int(model.nobs)}",
                            "Selected X Variables": ', '.join(selected_x_vars)
                        })

                    preview_df = pd.DataFrame(preview_data)
                    st.write(preview_df)

                # Save the full results in the session state for download
                st.session_state[f"{scenario_name}_results"] = scenario_results

                if st.button(f"Download Full Results for {scenario_name}"):
                    self.export_full_results(scenario_results, scenario_name)

    def export_full_results(self, results, scenario_name):
        full_data = []

        for result in results:
            output_df, selected_years, y_variable_name, model, anova_table, selected_x_vars, idx = result
            summary_data = []

            # Add selected years at the top
            summary_data.append(['Selected Years', ', '.join(map(str, selected_years))])

            # Add regression statistics
            summary_data.append(['SUMMARY OUTPUT', ''])
            summary_data.append([''])
            summary_data.append(['Regression Statistics', ''])
            summary_data.append(['Multiple R', f"{model.rsquared ** 0.5:.4f}"])
            summary_data.append(['R Square', f"{model.rsquared:.4f}"])
            summary_data.append(['Adjusted R Square', f"{model.rsquared_adj:.4f}"])
            summary_data.append(['Standard Error', f"{model.bse.mean():.4f}"])
            summary_data.append(['Observations', f"{int(model.nobs)}"])
            summary_data.append([''])

            # Add ANOVA table
            summary_data.append(['ANOVA', ''])
            summary_data.append(['', 'df', 'SS', 'MS', 'F', 'Significance F'])
            for index, row in anova_table.iterrows():
                summary_data.append([str(index)] + [str(item) if item is not None else '' for item in row.tolist()])
            summary_data.append([''])

            # Add coefficients
            summary_data.append(['', 'Coefficients', 'Standard Error', 't Stat', 'P-value', 'Lower 95%', 'Upper 95%'])
            coeff_table = pd.read_html(model.summary().as_html(), header=0, index_col=0)[1].reset_index()

            # Separate 'Constant' and other variables
            constant_row = coeff_table[coeff_table.iloc[:, 0] == 'const'].iloc[0].tolist()
            x_vars = coeff_table[coeff_table.iloc[:, 0] != 'const'].iloc[:, 0].tolist()

            # Sort remaining x variables alphabetically
            x_vars_sorted = sorted(x_vars)

            # Add 'Constant' first
            summary_data.append([str(item) if item is not None else '' for item in constant_row])

            # Add sorted x variables
            for var in x_vars_sorted:
                row = coeff_table[coeff_table.iloc[:, 0] == var].iloc[0].tolist()
                summary_data.append([str(item) if item is not None else '' for item in row])

            # Convert summary data to DataFrame
            summary_df = pd.DataFrame(summary_data)
            full_data.append(summary_df)

        # Combine all DataFrames into one Excel file
        with pd.ExcelWriter(f"{scenario_name}.xlsx", engine='xlsxwriter') as writer:
            for i, df in enumerate(full_data):
                df.to_excel(writer, sheet_name=f"Run {i+1}", index=False, header=False)

        with open(f"{scenario_name}.xlsx", 'rb') as f:
            data = f.read()
        st.download_button(label="Download Excel File", data=data, file_name=f"{scenario_name}.xlsx")

        # Clean up: delete the temporary Excel file
        os.remove(f"{scenario_name}.xlsx")

def main():
    st.set_page_config(layout="wide")

    app = RegressionApp()

    st.title("SG2024 Regression Analysis Crazy-Fast Tool")

    st.write("### Upload Xlsx Source File:")
    app.choose_file()

    if st.button("Run Regression Scenarios"):
        app.run_regression_scenarios()

    st.write("### Existing Scenarios:")
    app.display_scenarios()

    st.write("### Variables:")
    app.show_variable_selection()

    if "results" in st.session_state:
        app.display_results_page()

if __name__ == "__main__":
    main()
