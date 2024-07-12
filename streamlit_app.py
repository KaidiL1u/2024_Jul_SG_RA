import streamlit as st
import pandas as pd
import statsmodels.api as sm
import numpy as np
import itertools
import warnings
import pyperclip
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

            for i, selected_x_vars in enumerate(combinations, start=1):
                columns_to_keep = ['Year', self.df.columns[1]] + list(selected_x_vars)
                df_selected_sub = df_selected[columns_to_keep]
                model = self.run_regression(df_selected_sub)
                if model:
                    output_df = self.format_regression_output(model)
                    anova_table = self.calculate_anova_table(model)
                    scenario_results.append((output_df, years, self.df.columns[1], model, anova_table, selected_x_vars, i))

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

        tab_titles = [f"Scenario: {name}" for name, _ in all_results]
        tabs = st.tabs(tab_titles)

        for tab, (scenario_name, scenario_results) in zip(tabs, all_results):
            with tab:
                summary_data = []

                for result in scenario_results:
                    output_df, selected_years, y_variable_name, model, anova_table, selected_x_vars, combination_number = result

                    # Add selected years at the top
                    summary_data.append(['Selected Years', ', '.join(map(str, selected_years))])
                    summary_data.append(['SUMMARY OUTPUT', ''])
                    summary_data.append([''])
                    summary_data.append(['Regression Statistics', ''])
                    summary_data.append([f'S{combination_number}R^2', f"{model.rsquared:.4f}"])
                    summary_data.append([f'S{combination_number}SE', f"{model.scale ** 0.5:.4f}"])
                    summary_data.append(['Adjusted R Square', f"{model.rsquared_adj:.4f}"])
                    summary_data.append(['Standard Error of the Regression', f"{model.scale ** 0.5:.4f}"])
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
                    summary_data.append([f'S{combination_number}Const'] + [str(item) if item is not None else '' for item in constant_row[1:]])

                    # Add sorted x variables
                    for j, var in enumerate(x_vars_sorted, start=1):
                        row = coeff_table[coeff_table.iloc[:, 0] == var].iloc[0].tolist()
                        summary_data.append([f'S{combination_number}X{j}'] + [str(item) if item is not None else '' for item in row[1:]])

                    # Determine the number of blank rows to add
                    num_x_vars = len(selected_x_vars)
                    blank_rows_to_add = 20 - (15 + num_x_vars)
                    for _ in range(blank_rows_to_add):
                        summary_data.append([''] * 15)

                    # Add three blank rows between each output
                    for _ in range(3):
                        summary_data.append([''] * 15)

                summary_df = pd.DataFrame(summary_data)

                st.dataframe(summary_df.style.set_properties(**{'text-align': 'center'}).set_table_styles([dict(selector='th', props=[('text-align', 'center')])]))

                if st.button(f"Copy to Clipboard {scenario_name}"):
                    csv = summary_df.to_csv(sep='\t', index=False, header=False)
                    pyperclip.copy(csv)
                    st.success("Data copied to clipboard!")

                if st.button(f"Export {scenario_name} as Excel"):
                    self.export_excel(summary_df, scenario_name)

    def export_excel(self, df, scenario_name):
        # Create a Pandas Excel writer using XlsxWriter as the engine.
        excel_filename = f"{scenario_name}.xlsx"
        sheet_name = "Sheet1"
    
        # Save the dataframe to a writer object.
        with pd.ExcelWriter(excel_filename, engine='xlsxwriter') as writer:
            # Convert specific columns to numeric
            df.to_excel(writer, sheet_name=sheet_name, index=False)
    
            workbook = writer.book
            worksheet = writer.sheets[sheet_name]
    
            # Write headers and data while formatting numbers
            for col_num, value in enumerate(df.columns.values):
                worksheet.write(0, col_num, value)
    
            for col_num in range(1, len(df.columns)):  # Start from 1 to skip the first column (assumed to be non-numeric)
                column = df.columns[col_num]
                numeric_format = workbook.add_format({'num_format': '0.00'})  # Adjust '0.00' as needed
    
                for row_num in range(1, len(df) + 1):
                    cell_value = df.iloc[row_num - 1, col_num]
                    if isinstance(cell_value, (int, float)):
                        worksheet.write_number(row_num, col_num, cell_value, numeric_format)
                    else:
                        worksheet.write(row_num, col_num, cell_value)
    
        # Download the Excel file
        with open(excel_filename, 'rb') as f:
            data = f.read()
        st.download_button(label="Download Excel File", data=data, file_name=excel_filename)
    
        # Clean up: delete the temporary Excel file
        os.remove(excel_filename)


    @staticmethod
    def is_number(s):
        try:
            float(s)
            return True
        except ValueError:
            return False

    def run_regression(self, df):
        Y = df[self.df.columns[1]].astype(float)
        X = df[df.columns.difference(['Year', self.df.columns[1]])].astype(float)
        X = sm.add_constant(X)
        model = sm.OLS(Y, X).fit()
        return model

    def format_regression_output(self, model):
        summary_df = pd.read_html(model.summary().tables[1].as_html(), header=0, index_col=0)[0]
        return summary_df

    def calculate_anova_table(self, model):
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
            'Significance F': [f"{p_value:.4f}", np.nan, np.nan]
        }, index=['Regression', 'Residual', 'Total'])

        return anova_table

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
