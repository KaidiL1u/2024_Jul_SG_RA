import streamlit as st
import pandas as pd
import statsmodels.api as sm
import numpy as np
import itertools
import warnings
import pyperclip  # Ensure pyperclip is installed
import os

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

class RegressionApp:
    def __init__(self):
        self.df = None
        self.variables = []
        self.scenarios = dict(predefined_years)
        self.num_combinations = 0

    def choose_file(self):
        file = st.file_uploader("Upload Excel file", type=["xlsx"])
        if file:
            self.df = pd.read_excel(file)
            self.variables = self.df.columns[2:].tolist()  # Assuming variables start from column C onwards

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

        st.dataframe(scenario_df.T.style.set_properties(**{'text-align': 'center'}).set_table_styles([dict(selector='th', props=[('text-align', 'center')])]))

    def run_regression_scenarios(self):
        if self.df is None:
            st.warning("Please upload an Excel file first.")
            return

        all_results = []

        for scenario_name, years in self.scenarios.items():
            if not years:  # If years selection is empty, use predefined years
                years = predefined_years[scenario_name]

            df_selected = self.df[self.df['Year'].isin(years)]
            variables = self.df.columns[2:].tolist()  # Assuming variables start from column C onwards
            num_variables = len(variables)

            combinations = itertools.chain.from_iterable(
                itertools.combinations(variables, r) for r in range(1, num_variables + 1)
            )

            scenario_results = []

            for idx, selected_x_vars in enumerate(combinations, start=1):
                columns_to_keep = ['Year', self.df.columns[1]] + list(selected_x_vars)
                df_selected_sub = df_selected[columns_to_keep]
                model = self.run_regression(df_selected_sub)
                if model:
                    output_df = self.format_regression_output(model)
                    anova_table = self.calculate_anova_table(model)
                    scenario_results.append((output_df, years, self.df.columns[1], model, anova_table, selected_x_vars, idx))

            all_results.append((scenario_name, scenario_results))

        self.show_combined_results_window(all_results)

    def show_combined_results_window(self, all_results):
        st.session_state["results"] = all_results
        st.experimental_rerun()

    def display_results_page(self):
        if "results" not in st.session_state:
            st.write("No results to display. Please run the regression scenarios first.")
            return

        all_results = st.session_state["results"]

        # Prepare to display up to 5 scenarios
        num_tabs = min(5, len(all_results))
        tab_titles = [f"Scenario: {name}" for name, _ in all_results[:num_tabs]]
        tabs = st.tabs(tab_titles)

        for tab, (scenario_name, scenario_results) in zip(tabs, all_results[:num_tabs]):
            with tab:
                summary_data = []

                for result in scenario_results:
                    output_df, selected_years, y_variable_name, model, anova_table, selected_x_vars, idx = result

                    # Add selected years at the top
                    summary_data.append(['', 'Selected Years', ', '.join(map(str, selected_years))])
                    summary_data.append(['', 'SUMMARY OUTPUT', ''])
                    summary_data.append([''])
                    summary_data.append(['', 'Regression Statistics', ''])
                    summary_data.append(['', 'Multiple R', f"{model.rsquared ** 0.5:.4f}"])
                    summary_data.append([f"S{idx}R^2", 'R Square', f"{model.rsquared:.4f}"])
                    summary_data.append(['', 'Adjusted R Square', f"{model.rsquared_adj:.4f}"])
                    summary_data.append([f"S{idx}SE", 'Standard Error of the Regression', f"{model.scale ** 0.5:.4f}"])
                    summary_data.append(['', 'Observations', f"{int(model.nobs)}"])
                    summary_data.append([''])

                    # Add ANOVA table
                    summary_data.append(['', 'ANOVA', ''])
                    summary_data.append(['', '', 'df', 'SS', 'MS', 'F', 'Significance F'])
                    for index, row in anova_table.iterrows():
                        summary_data.append(['', str(index)] + [str(item) if item is not None else '' for item in row.tolist()])
                    summary_data.append([''])

                    # Add coefficients if available
                    if hasattr(model, 'summary') and len(model.summary().tables) > 1:
                        coeff_table = pd.read_html(model.summary().tables[1].as_html(), header=0, index_col=0)[0].reset_index()
                        summary_data.append(['', '', 'Coefficients', 'Standard Error', 't Stat', 'P-value', 'Lower 95%', 'Upper 95%'])

                        # Separate 'Constant' and other variables
                        constant_row = coeff_table[coeff_table.iloc[:, 0] == 'const'].iloc[0].tolist()
                        x_vars = coeff_table[coeff_table.iloc[:, 0] != 'const'].iloc[:, 0].tolist()

                        # Sort remaining x variables alphabetically
                        x_vars_sorted = sorted(x_vars)

                        # Add 'Constant' first
                        summary_data.append([f"S{idx}Const"] + [str(item) if item is not None else '' for item in constant_row])

                        # Add sorted x variables
                        for i, var in enumerate(x_vars_sorted, start=1):
                            row = coeff_table[coeff_table.iloc[:, 0] == var].iloc[0].tolist()
                            summary_data.append([f"S{idx}X{i}"] + [str(item) if item is not None else '' for item in row])

                        # Determine the number of blank rows to add
                        num_x_vars = len(selected_x_vars)
                        blank_rows_to_add = 20 - (10 + num_x_vars)
                        for _ in range(blank_rows_to_add):
                            summary_data.append([''] * 10)

                        # Add x no.of blank rows between each output
                        for _ in range(2): # replace the number in here as x
                            summary_data.append([''] * 10)

                summary_df = pd.DataFrame(summary_data)

                st.dataframe(summary_df.style.set_properties(**{'text-align': 'center'}).set_table_styles([dict(selector='th', props=[('text-align', 'center')])]))

                if st.button(f"Copy to Clipboard {scenario_name}"):
                    try:
                        csv = summary_df.to_csv(sep='\t', index=False, header=False)
                        pyperclip.copy(csv)
                        st.success("Data copied to clipboard!")
                    except pyperclip.PyperclipException as e:
                        st.error("Copying to clipboard failed. Please use another method to copy the data.")

                if st.button(f"Export {scenario_name} as Excel"):
                    self.export_excel(summary_df, scenario_name)

    def run_regression(self, df):
        # Replace with your own regression logic based on the dataframe `df`
        try:
            y = df[df.columns[1]]
            X = df[df.columns[2:]]
            X = sm.add_constant(X)
            model = sm.OLS(y, X).fit()
            return model
        except Exception as e:
            st.error(f"Error occurred during regression: {str(e)}")
            return None

    def format_regression_output(self, model):
        # Replace with your own formatting logic based on the regression `model`
        output_df = pd.DataFrame({
            "Coefficient": model.params,
            "Standard Error": model.bse,
            "t value": model.tvalues,
            "P value": model.pvalues
        })
        return output_df

    def calculate_anova_table(self, model):
        # Replace with your own ANOVA calculation logic based on the regression `model`
        anova_table = sm.stats.anova_lm(model, typ=2)
        return anova_table

    def export_excel(self, df, scenario_name):
        # Replace with your own logic to export dataframe `df` to Excel
        file_path = f"{scenario_name}_results.xlsx"
        df.to_excel(file_path, index=False)
        st.success(f"Data exported successfully to {file_path}")

    def run(self):
        st.title("Regression Analysis App")

        self.choose_file()

        if self.df is not None:
            st.write("### File Uploaded Successfully!")
            self.show_variable_selection()
            self.display_scenarios()

            if st.button("Run Regression Scenarios"):
                self.run_regression_scenarios()

            self.display_results_page()

if __name__ == "__main__":
    app = RegressionApp()
    app.run()
