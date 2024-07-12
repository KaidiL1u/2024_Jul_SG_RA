import streamlit as st
import pandas as pd
import statsmodels.api as sm
import numpy as np
import itertools
import warnings
import pyperclip
import os
import platform

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

            # Counter for scenario-specific labels
            scenario_label_counter = 1

            for selected_x_vars in combinations:
                columns_to_keep = ['Year', self.df.columns[1]] + list(selected_x_vars)
                df_selected_sub = df_selected[columns_to_keep]
                model = self.run_regression(df_selected_sub)
                if model:
                    output_df = self.format_regression_output(model)
                    anova_table = self.calculate_anova_table(model)

                    # Adding scenario-specific labels
                    summary_data = self.add_scenario_labels(output_df, selected_x_vars, model, anova_table, scenario_label_counter)
                    scenario_results.append(summary_data)

                    scenario_label_counter += 1

            all_results.append((scenario_name, scenario_results))

        self.show_combined_results_window(all_results)

    def add_scenario_labels(self, output_df, selected_x_vars, model, anova_table, scenario_label_counter):
        summary_data = []

        # Scenario labels
        s_r_square_label = f"S{scenario_label_counter}R^2"
        s_se_label = f"S{scenario_label_counter}SE"
        s_const_label = f"S{scenario_label_counter}const"

        # Add selected years at the top
        summary_data.append(['Selected Years', ''])
        # Add S1R^2
        summary_data.append([s_r_square_label, 'R Square'])
        summary_data.append([''])
        # Add S1SE
        summary_data.append([s_se_label, 'Standard Error of the Regression'])
        summary_data.append([''])
        # Add S1const
        summary_data.append([s_const_label, 'const'])

        # Add X variables labels
        x_vars_sorted = sorted(selected_x_vars)
        x_label_counter = 1
        for var in x_vars_sorted:
            s_x_label = f"S{scenario_label_counter}X{x_label_counter}"
            summary_data.append([s_x_label, var])
            x_label_counter += 1

        # Append the rest of the summary data
        summary_data.extend([
            ['', 'Regression Statistics'],
            ['Multiple R', f"{model.rsquared ** 0.5:.4f}"],
            ['R Square', f"{model.rsquared:.4f}"],
            ['Adjusted R Square', f"{model.rsquared_adj:.4f}"],
            ['Standard Error of the Regression', f"{model.scale ** 0.5:.4f}"],
            ['Observations', f"{int(model.nobs)}"],
            [''],
            ['ANOVA', ''],
        ])

        # Add ANOVA table
        for index, row in anova_table.iterrows():
            summary_data.append(['', str(index)] + [str(item) if item is not None else '' for item in row.tolist()])

        # Add blank rows for spacing
        summary_data.append(['', ''])

        return summary_data

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
                for i, result in enumerate(scenario_results):
                    summary_data = pd.DataFrame(result)
    
                    st.dataframe(summary_data.style.set_properties(**{'text-align': 'center'}).set_table_styles([dict(selector='th', props=[('text-align', 'center')])]))
    
                    # Use a unique widget key for each button based on scenario_name and index i
                    if st.button(f"Copy to Clipboard {scenario_name} - {i}", key=f"copy_button_{scenario_name}_{i}"):
                        self.copy_to_clipboard(summary_data)
                        st.success("Data copied to clipboard!")
    
                    # Use a unique widget key for each button based on scenario_name and index i
                    if st.button(f"Export {scenario_name} as Excel - {i}", key=f"export_button_{scenario_name}_{i}"):
                        self.export_excel(summary_data, scenario_name)


    def copy_to_clipboard(self, df):
        try:
            # Convert DataFrame to tab-separated format for compatibility across platforms
            df_str = df.to_csv(sep='\t', index=False, header=False)
            pyperclip.copy(df_str)
        except Exception as e:
            st.error(f"Copying to clipboard failed: {e}")

    def export_excel(self, df, scenario_name):
        # Create a Pandas Excel writer using XlsxWriter as the engine.
        excel_filename = f"{scenario_name}.xlsx"
        sheet_name = "Sheet1"

        # Save the dataframe to a writer object.
        with pd.ExcelWriter(excel_filename, engine='xlsxwriter') as writer:
            df.to_excel(writer, sheet_name=sheet_name, index=False)

        # Download the Excel file
        with open(excel_filename, 'rb') as f:
            data = f.read()
        st.download_button(label="Download Excel File", data=data, file_name=excel_filename)

        # Clean up: delete the temporary Excel file
        os.remove(excel_filename)

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
