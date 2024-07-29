import streamlit as st
import pandas as pd
import statsmodels.api as sm
import numpy as np
import itertools
import warnings
import os
import time

# Suppress warnings about future changes
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

# Predefined year selections
predefined_years = {
    "2001-2023": list(range(2001, 2024)),
    "2005 - 2023 excluding 2009, 2016": [year for year in range(2005, 2024) if year not in {2009, 2016}],
    "2001-2023 excluding 2020, 2021, 2022": [year for year in range(2001, 2024) if year not in {2020, 2021, 2022}],
    "2001-2019": list(range(2001, 2020)),
    "2001-2023 excluding 2009, 2013, 2020, 2021, 2022": [
        year for year in range(2001, 2024) if year not in {2009, 2013, 2020, 2021, 2022}
    ]
}

def run_single_regression(df, y_var, x_vars):
    Y = df[y_var].astype(float)
    X = df[x_vars].astype(float)
    X = sm.add_constant(X)
    model = sm.OLS(Y, X).fit()
    return model

class RegressionApp:
    def __init__(self):
        self.df = None
        self.variables = []
        self.scenarios = dict(predefined_years)
        self.num_combinations = 0
        self.total_regressions = 0
        self.completed_regressions = 0
        self.start_time = None

    def choose_file(self):
        file = st.file_uploader("Upload Excel file", type=["xlsx"])
        if file:
            self.df = pd.read_excel(file, sheet_name="Sheet1")
            self.variables = self.df.columns[2:].tolist()
            st.write("### Columns in the uploaded file:")
            st.write(self.df.columns.tolist())

    def show_variable_selection(self):
        if self.df is None:
            st.warning("Please upload an Excel file first.")
            return

        self.variables = self.df.columns[2:].tolist()
        self.num_combinations = sum(
            [len(list(itertools.combinations(self.variables, i))) for i in range(1, len(self.variables) + 1)])
        self.total_regressions = len(self.scenarios) * self.num_combinations

        if self.total_regressions == 0:
            st.warning("No regression scenarios to run based on the current data and setup.")
        else:
            st.subheader(f"{len(self.variables)} variables found.")
            st.subheader(f"Regression will create {self.num_combinations} variable combinations "
                         f"(Total Regressions: {self.total_regressions}).")
            st.write("### Variables:")
            for var in self.variables:
                st.write(f"- {var}")

    def display_scenarios(self):
        all_years = list(range(2001, 2024))
        scenario_df = pd.DataFrame(columns=all_years)

        for name, years in self.scenarios.items():
            scenario_df.loc[name] = ['•' if year in years else '' for year in all_years]

        st.table(scenario_df.T)

    def run_regression_scenarios(self):
        if self.df is None:
            st.warning("Please upload an Excel file first.")
            return

        if 'Year' not in self.df.columns:
            st.error("The 'Year' column is missing from the uploaded file.")
            return

        all_results = []
        self.start_time = time.time()
        progress_bar = st.progress(0)
        progress_text = st.empty()

        for scenario_name, years in self.scenarios.items():
            df_selected = self.df[self.df['Year'].isin(years)]
            variables = self.df.columns[2:].tolist()
            combinations = list(itertools.chain.from_iterable(
                itertools.combinations(variables, r) for r in range(1, len(variables) + 1)
            ))

            scenario_results = []
            for idx, selected_x_vars in enumerate(combinations, start=1):
                model = run_single_regression(df_selected, self.df.columns[1], list(selected_x_vars))
                output_df = self.format_regression_output(model)
                if output_df.empty:
                    st.warning(f"Could not compute output for model with variables {selected_x_vars}")
                    continue
                anova_table = self.calculate_anova_table(model)
                scenario_results.append(
                    (output_df, years, self.df.columns[1], model, anova_table, selected_x_vars, idx))
                self.completed_regressions += 1
                self.update_progress(progress_bar, progress_text)

            all_results.append((scenario_name, scenario_results))

        st.session_state["results"] = all_results

    def update_progress(self, progress_bar, progress_text):
        if self.total_regressions > 0:
            progress_percent = self.completed_regressions / self.total_regressions
            elapsed_time = time.time() - self.start_time
            estimated_total_time = elapsed_time * self.total_regressions / self.completed_regressions
            time_left = estimated_total_time - elapsed_time

            progress_bar.progress(progress_percent)
            progress_text.text(f"Completed {self.completed_regressions} out of {self.total_regressions} regressions. "
                               f"Time left: {time_left:.2f} seconds. Records left to run: "
                               f"{self.total_regressions - self.completed_regressions}.")
        else:
            progress_bar.progress(0)
            progress_text.text("No regressions to run.")

    def display_results_page(self):
        if "results" not in st.session_state:
            st.write("No results to display. Please run the regression scenarios first.")
            return

        all_results = st.session_state["results"]
        num_tabs = min(5, len(all_results))
        tab_titles = [f"Scenario: {name}" for name, _ in all_results[:num_tabs]]
        tabs = st.tabs(tab_titles)

        for tab, (scenario_name, scenario_results) in zip(tabs, all_results[:num_tabs]):
            with tab:
                summary_data = []
                for result in scenario_results:
                    output_df, selected_years, y_variable_name, model, anova_table, selected_x_vars, idx = result
                    summary_data.append(['', 'Selected Years', ', '.join(map(str, selected_years))])
                    summary_data.append(['', 'SUMMARY OUTPUT', ''])
                    summary_data.append(['', 'Regression Statistics', ''])
                    summary_data.append(['', 'Multiple R', f"{model.rsquared ** 0.5:.4f}"])
                    summary_data.append([f"S{idx}R^2", 'R Square', f"{model.rsquared:.4f}"])
                    summary_data.append(['', 'Adjusted R Square', f"{model.rsquared_adj:.4f}"])
                    summary_data.append([f"S{idx}SE", 'Standard Error of the Regression', f"{model.scale ** 0.5:.4f}"])
                    summary_data.append(['', 'Observations', f"{int(model.nobs)}"])
                    summary_data.append(['', 'ANOVA', ''])
                    summary_data.append(['', '', 'df', 'SS', 'MS', 'F', 'Significance F'])
                    for index, row in anova_table.iterrows():
                        summary_data.append(['', str(index)] + [str(item) if item is not None else '' for item in row.tolist()])

                    coeff_table = pd.read_html(model.summary().tables[1].as_html(), header=0, index_col=0)[0].reset_index()
                    summary_data.append(
                        ['', '', 'Coefficients', 'Standard Error', 't Stat', 'P-value', 'Lower 95%', 'Upper 95%'])
                    constant_row = coeff_table[coeff_table.iloc[:, 0] == 'const'].iloc[0].tolist()
                    x_vars = coeff_table[coeff_table.iloc[:, 0] != 'const'].iloc[:, 0].tolist()
                    x_vars_sorted = sorted(x_vars)
                    summary_data.append([f"S{idx}Const"] + [str(item) if item is not None else '' for item in constant_row])
                    for i, var in enumerate(x_vars_sorted, start=1):
                        row = coeff_table[coeff_table.iloc[:, 0] == var].iloc[0].tolist()
                        summary_data.append([f"S{idx}X{i}"] + [str(item) if item is not None else '' for item in row])

                summary_df = pd.DataFrame(summary_data)
                st.dataframe(summary_df)
                if st.button(f"Copy to Clipboard {scenario_name}"):
                    csv = summary_df.to_csv(sep='\t', index=False, header=False)
                    st.session_state[f"{scenario_name}_csv"] = csv
                    st.success("Data prepared for clipboard copying. Click the button below to copy.")
                    if st.button("Copy Now"):
                        st.write(f"Copy the data manually from here:\n\n{csv}\n")

                if st.button(f"Export {scenario_name} as Excel"):
                    self.export_excel(summary_df, scenario_name)

    def export_excel(self, df, scenario_name):
        excel_filename = f"{scenario_name}.xlsx"
        sheet_name = "Sheet1"
        with pd.ExcelWriter(excel_filename, engine='xlsxwriter') as writer:
            df.to_excel(writer, sheet_name=sheet_name, index=False)
        with open(excel_filename, 'rb') as f:
            data = f.read()
        st.download_button(label="Download Excel File", data=data, file_name=excel_filename)
        os.remove(excel_filename)

    def format_regression_output(self, model):
        try:
            summary_html = model.summary().tables[1].as_html()
            summary_df = pd.read_html(summary_html, header=0, index_col=0)[0]
            return summary_df
        except Exception as e:
            st.error("Failed to format regression output: " + str(e))
            return pd.DataFrame()  # Return an empty DataFrame on failure

    def calculate_anova_table(self, model):
        sse = model.ssr
        ssr = model.ess
        sst = ssr + sse
        dfe = model.df_resid
        dfr = model.df_model
        dft = dfr + dfe
        mse = sse / dfe
        msr = ssr / dfr
        f_stat = msr / mse
        p_value = model.f_pvalue
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
    st.title("SG2024 Regression Analysis Tool")
    app.choose_file()
    if st.button("Run Regression Scenarios"):
        with st.spinner("Running regression scenarios..."):
            app.run_regression_scenarios()
    app.display_scenarios()
    app.show_variable_selection()
    if "results" in st.session_state:
        app.display_results_page()

if __name__ == "__main__":
    main()
