# Stage-1 output inventory (2026-10-24)

Every `savefig`, `write_html`, `to_csv`, `to_excel` call in the scripts being consolidated (read at the working tree after commit 8e05544). The new `script1a`–`script1d` must cover each row or the row is marked dropped with a reason. File names built from variables are listed with the call line.

## Counts by script

| Script | figures | maps/html | csv tables | xlsx |
|---|---|---|---|---|
| `script1a-var-completeness.py` | 3 | 0 | 12 | 0 |
| `script2a-panel-qa.py` | 19 | 0 | 27 | 0 |
| `script2b-final-visuals.py` | 11 | 2 | 2 | 0 |
| `script2c-cafo-composition.py` | 5 | 0 | 0 | 0 |
| `script2d-threshold-presence.py` | 4 | 0 | 0 | 0 |
| `script2e-hispanic-chr-explorer.py` | 4 | 0 | 1 | 0 |
| `script2f-maps.py` | 3 | 3 | 1 | 0 |
| `script4s-species-overlap.py` | 1 | 0 | 6 | 0 |
| `script4v-trend-plots.py` | 3 | 0 | 3 | 0 |
| `script4k-variable-register.py` | 0 | 0 | 1 | 3 |

## Calls

| Script | Line | Section | Kind | File name or pattern |
|---|---|---|---|---|
| `script1a-var-completeness.py` | 98 |  | table (csv) | `qa_panel_integrity_mental_focus.csv` |
| `script1a-var-completeness.py` | 114 |  | table (csv) | `qa_panel_source_duplication_counts.csv` |
| `script1a-var-completeness.py` | 143 |  | table (csv) | `qa_mental_focus_variable_inventory.csv` |
| `script1a-var-completeness.py` | 172 |  | table (csv) | `qa_mental_outcome_core_overall.csv` |
| `script1a-var-completeness.py` | 185 |  | table (csv) | `qa_mental_outcome_counties_any.csv` |
| `script1a-var-completeness.py` | 197 |  | table (csv) | `qa_mental_outcome_core_by_year.csv` |
| `script1a-var-completeness.py` | 265 |  | table (csv) | `qa_mental_crosssource_duplicate_check.csv` |
| `script1a-var-completeness.py` | 374 | SECTION 6: QA VISUALIZATIONS | figure | `(name built from a variable: fig.savefig(heatmap_path, dpi=200, bbox_inches="tight"))` |
| `script1a-var-completeness.py` | 376 | SECTION 6: QA VISUALIZATIONS | table (csv) | `{today_str}_qa_coverage_matrix.csv` |
| `script1a-var-completeness.py` | 410 | SECTION 6: QA VISUALIZATIONS | table (csv) | `{today_str}_qa_panel_balance_by_year.csv` |
| `script1a-var-completeness.py` | 411 | SECTION 6: QA VISUALIZATIONS | table (csv) | `{today_str}_qa_panel_balance_summary.csv` |
| `script1a-var-completeness.py` | 445 | SECTION 6: QA VISUALIZATIONS | table (csv) | `(name built from a variable: supp_df.to_csv(supp_path, index=False))` |
| `script1a-var-completeness.py` | 504 | SECTION 6: QA VISUALIZATIONS | figure | `(name built from a variable: fig.savefig(dist_path, dpi=200, bbox_inches="tight"))` |
| `script1a-var-completeness.py` | 563 | SECTION 6: QA VISUALIZATIONS | figure | `(name built from a variable: fig.savefig(cafo_dist_path, dpi=200, bbox_inches="tight"))` |
| `script1a-var-completeness.py` | 577 | SECTION 6: QA VISUALIZATIONS | table (csv) | `(name built from a variable: outlier_df.to_csv(outlier_path, index=False))` |
| `script2a-panel-qa.py` | 120 |  | table (csv) | `(name built from a variable: obj.to_csv(path, index=False))` |
| `script2a-panel-qa.py` | 196 |  | table (csv) | `(name built from a variable: df.loc[dup_mask].to_csv()` |
| `script2a-panel-qa.py` | 355 | SECTION 1 — CDC crude-rate sense check | table (csv) | `{today_str}_qa_cdc_cruderate_sensecheck_overall.csv` |
| `script2a-panel-qa.py` | 356 | SECTION 1 — CDC crude-rate sense check | table (csv) | `{today_str}_qa_cdc_cruderate_sensecheck_by_year.csv` |
| `script2a-panel-qa.py` | 357 | SECTION 1 — CDC crude-rate sense check | table (csv) | `{today_str}_qa_cdc_cruderate_sensecheck_outliers_top1000.csv` |
| `script2a-panel-qa.py` | 394 | SECTION 2 — Correlation tables + FSIS column QA + QA memo | table (csv) | `correlation_vars_manifest.csv` |
| `script2a-panel-qa.py` | 414 | SECTION 2 — Correlation tables + FSIS column QA + QA memo | table (csv) | `qa_key_variable_coverage_by_year.csv` |
| `script2a-panel-qa.py` | 436 | SECTION 2 — Correlation tables + FSIS column QA + QA memo | table (csv) | `qa_fsis_12_column_check.csv` |
| `script2a-panel-qa.py` | 471 | SECTION 2 — Correlation tables + FSIS column QA + QA memo | table (csv) | `correlation_long_{wname}_{method}.csv` |
| `script2a-panel-qa.py` | 472 | SECTION 2 — Correlation tables + FSIS column QA + QA memo | table (csv) | `correlation_top30_{wname}_{method}.csv` |
| `script2a-panel-qa.py` | 473 | SECTION 2 — Correlation tables + FSIS column QA + QA memo | table (csv) | `(name built from a variable: sub_num[use_cols].corr(method=method, min_periods=MIN_CORR_N)` |
| `script2a-panel-qa.py` | 536 | SECTION 3 — FSIS size bins vs. poor mental health days (2017 | table (csv) | `{today_str}_fsis_size_bins_vs_pmhd_2017_plotdata.csv` |
| `script2a-panel-qa.py` | 548 | SECTION 3 — FSIS size bins vs. poor mental health days (2017 | figure | `(name built from a variable: g.savefig(fig_path, dpi=240, bbox_inches="tight"))` |
| `script2a-panel-qa.py` | 597 | SECTION 4 — Missingness tables, heatmaps, row completeness | figure | `{today_str}_missing_share_by_year.png` |
| `script2a-panel-qa.py` | 608 | SECTION 4 — Missingness tables, heatmaps, row completeness | figure | `(name built from a variable: plt.savefig(p, dpi=220, bbox_inches="tight"))` |
| `script2a-panel-qa.py` | 624 | SECTION 4 — Missingness tables, heatmaps, row completeness | figure | `{today_str}_missingness_{safe_name(family)}_chunk{chunk_idx}.png` |
| `script2a-panel-qa.py` | 634 | SECTION 4 — Missingness tables, heatmaps, row completeness | figure | `{today_str}_row_completeness_hist_all.png` |
| `script2a-panel-qa.py` | 641 | SECTION 4 — Missingness tables, heatmaps, row completeness | figure | `{today_str}_row_completeness_box_by_year.png` |
| `script2a-panel-qa.py` | 733 | SECTION 6 — Violin plots: high-coverage variables | figure | `{today_str}_violin_{safe_name(var)}.png` |
| `script2a-panel-qa.py` | 762 | SECTION 7 — CAFO composition from compact | figure | `{today_str}_cafo_violin_{commodity}.png` |
| `script2a-panel-qa.py` | 770 | SECTION 7 — CAFO composition from compact | figure | `{today_str}_cafo_violin_commodity_compare.png` |
| `script2a-panel-qa.py` | 786 | SECTION 7 — CAFO composition from compact | table (csv) | `{today_str}_size_commodity_share.csv` |
| `script2a-panel-qa.py` | 808 | SECTION 7 — CAFO composition from compact | figure | `{today_str}_share_commodity_within_{size}.png` |
| `script2a-panel-qa.py` | 818 | SECTION 7 — CAFO composition from compact | table (csv) | `{today_str}_commodity_size_share.csv` |
| `script2a-panel-qa.py` | 840 | SECTION 7 — CAFO composition from compact | figure | `{today_str}_share_size_within_{commodity}.png` |
| `script2a-panel-qa.py` | 869 | SECTION 8 — CAFO concentration: HHI, Gini, entry/exit by yea | table (csv) | `{today_str}_cafo_concentration_by_year_commodity.csv` |
| `script2a-panel-qa.py` | 880 | SECTION 8 — CAFO concentration: HHI, Gini, entry/exit by yea | figure | `{today_str}_cafo_concentration_{metric}.png` |
| `script2a-panel-qa.py` | 900 | SECTION 8 — CAFO concentration: HHI, Gini, entry/exit by yea | table (csv) | `{today_str}_cafo_transition_matrix_overall.csv` |
| `script2a-panel-qa.py` | 903 | SECTION 8 — CAFO concentration: HHI, Gini, entry/exit by yea | table (csv) | `{today_str}_cafo_transition_matrix_by_year.csv` |
| `script2a-panel-qa.py` | 913 | SECTION 8 — CAFO concentration: HHI, Gini, entry/exit by yea | table (csv) | `{today_str}_cafo_entry_exit_large_by_year.csv` |
| `script2a-panel-qa.py` | 919 | SECTION 8 — CAFO concentration: HHI, Gini, entry/exit by yea | figure | `{today_str}_cafo_entry_exit_rates.png` |
| `script2a-panel-qa.py` | 957 | SECTION 9 — Ridge plots + trend plots + binned scatter (slow | figure | `{today_str}_ridge_{safe_name(var)}.png` |
| `script2a-panel-qa.py` | 977 | SECTION 9 — Ridge plots + trend plots + binned scatter (slow | table (csv) | `(name built from a variable: pd.DataFrame({"variable": outcome_vars}).to_csv()` |
| `script2a-panel-qa.py` | 1011 | SECTION 9 — Ridge plots + trend plots + binned scatter (slow | figure | `{today_str}_trend_{safe_name(commodity)}_{safe_name(var)}.png` |
| `script2a-panel-qa.py` | 1049 | SECTION 9 — Ridge plots + trend plots + binned scatter (slow | figure | `(name built from a variable: plt.savefig(os.path.join(binned_dir,)` |
| `script2a-panel-qa.py` | 1070 | SECTION 10 — State-level CAFO trends | table (csv) | `{today_str}_states_with_large_cafo.csv` |
| `script2a-panel-qa.py` | 1111 | SECTION 10 — State-level CAFO trends | figure | `(name built from a variable: fig.savefig(os.path.join(state_dir,)` |
| `script2a-panel-qa.py` | 1169 | SECTION 11 — Targeted sumstats: county coverage check, CAFO/ | table (csv) | `county_coverage_check.csv` |
| `script2a-panel-qa.py` | 1173 | SECTION 11 — Targeted sumstats: county coverage check, CAFO/ | table (csv) | `(name built from a variable: summarize_numeric(df, cafo_cols_panel).to_csv()` |
| `script2a-panel-qa.py` | 1176 | SECTION 11 — Targeted sumstats: county coverage check, CAFO/ | table (csv) | `(name built from a variable: summarize_numeric(state_year_cafo, cafo_cols_panel).to_csv()` |
| `script2a-panel-qa.py` | 1182 | SECTION 11 — Targeted sumstats: county coverage check, CAFO/ | table (csv) | `(name built from a variable: summarize_numeric(df_fsis, [c for c in fsis_panel_cols if c )` |
| `script2a-panel-qa.py` | 1193 | SECTION 11 — Targeted sumstats: county coverage check, CAFO/ | table (csv) | `mental_fill_share.csv` |
| `script2a-panel-qa.py` | 1241 | SECTION 11 — Targeted sumstats: county coverage check, CAFO/ | table (csv) | `(name built from a variable: pd.DataFrame(cross_rows).to_csv()` |
| `script2a-panel-qa.py` | 1270 | SECTION 12 — State dot plots + chickens size-class facets | figure | `state_county_dots_{st_lbl}_2017.png` |
| `script2a-panel-qa.py` | 1293 | SECTION 12 — State dot plots + chickens size-class facets | figure | `chickens_cafo_vs_poor_mental_health_faceted_2010_2020.png` |
| `script2a-panel-qa.py` | 1298 | SECTION 12 — State dot plots + chickens size-class facets | table (csv) | `(name built from a variable: "chicken_size_class"]].to_csv()` |
| `script2b-final-visuals.py` | 250 |  | figure | `(name built from a variable: fig.savefig(a1_path, dpi=200, bbox_inches="tight"))` |
| `script2b-final-visuals.py` | 301 |  | figure | `(name built from a variable: fig.savefig(a2_path, dpi=200, bbox_inches="tight"))` |
| `script2b-final-visuals.py` | 347 |  | figure | `(name built from a variable: fig.savefig(a3_path, dpi=200, bbox_inches="tight"))` |
| `script2b-final-visuals.py` | 459 |  | figure | `(name built from a variable: fig.savefig(b_path, dpi=200, bbox_inches="tight"))` |
| `script2b-final-visuals.py` | 469 |  | table (csv) | `(name built from a variable: selection_meta.to_csv()` |
| `script2b-final-visuals.py` | 515 |  | figure | `(name built from a variable: fig.savefig(c_path, dpi=200, bbox_inches="tight"))` |
| `script2b-final-visuals.py` | 517 |  | table (csv) | `{today_str}_C_outcome_crosscorrelation.csv` |
| `script2b-final-visuals.py` | 594 |  | figure | `(name built from a variable: fig.savefig(d_path, dpi=200, bbox_inches="tight"))` |
| `script2b-final-visuals.py` | 648 |  | figure | `(name built from a variable: fig.savefig(e_path, dpi=200, bbox_inches="tight"))` |
| `script2b-final-visuals.py` | 720 |  | map/html | `(name built from a variable: _fig_map.write_html(_html_path))` |
| `script2b-final-visuals.py` | 726 |  | figure (png) | `(name built from a variable: _fig_map.write_image(_png_path, width=1400, height=800, scal)` |
| `script2b-final-visuals.py` | 798 |  | map/html | `(name built from a variable: fig_ch.write_html(_html))` |
| `script2b-final-visuals.py` | 801 |  | figure (png) | `(name built from a variable: fig_ch.write_image(_png, width=430 * _n_cols, height=360 * _)` |
| `script2b-final-visuals.py` | 903 |  | figure | `(name built from a variable: _fig_i.savefig(_i_path, dpi=200, bbox_inches="tight"))` |
| `script2b-final-visuals.py` | 964 |  | figure | `(name built from a variable: _fig_j.savefig(_j_path, dpi=200, bbox_inches="tight"))` |
| `script2c-cafo-composition.py` | 119 |  | figure | `(name built from a variable: fig.savefig(path, dpi=200, bbox_inches="tight"))` |
| `script2c-cafo-composition.py` | 165 |  | figure | `(name built from a variable: fig.savefig(path, dpi=200, bbox_inches="tight"))` |
| `script2c-cafo-composition.py` | 229 |  | figure | `(name built from a variable: fig.savefig(path, dpi=200, bbox_inches="tight"))` |
| `script2c-cafo-composition.py` | 328 |  | figure | `(name built from a variable: fig.savefig(path, dpi=200, bbox_inches="tight"))` |
| `script2c-cafo-composition.py` | 432 |  | figure | `(name built from a variable: fig.savefig(path_p, dpi=200, bbox_inches="tight"))` |
| `script2d-threshold-presence.py` | 215 |  | figure | `(name built from a variable: fig_o.savefig(o_path, dpi=200, bbox_inches="tight"))` |
| `script2d-threshold-presence.py` | 260 |  | figure | `(name built from a variable: fig_p.savefig(p_path, dpi=200, bbox_inches="tight"))` |
| `script2d-threshold-presence.py` | 343 |  | figure | `(name built from a variable: fig_q.savefig(q_path, dpi=200, bbox_inches="tight"))` |
| `script2d-threshold-presence.py` | 402 |  | figure | `(name built from a variable: fig_r.savefig(r_path, dpi=200, bbox_inches="tight"))` |
| `script2e-hispanic-chr-explorer.py` | 193 |  | figure | `(name built from a variable: fig.savefig(path, dpi=200, bbox_inches="tight"))` |
| `script2e-hispanic-chr-explorer.py` | 242 |  | figure | `(name built from a variable: fig.savefig(path, dpi=200, bbox_inches="tight"))` |
| `script2e-hispanic-chr-explorer.py` | 318 |  | figure | `(name built from a variable: fig.savefig(path, dpi=200, bbox_inches="tight"))` |
| `script2e-hispanic-chr-explorer.py` | 417 |  | figure | `(name built from a variable: fig.savefig(path, dpi=200, bbox_inches="tight"))` |
| `script2e-hispanic-chr-explorer.py` | 446 |  | table (csv) | `(name built from a variable: stats_df.to_csv(csv_path, index=False))` |
| `script2f-maps.py` | 142 | SECTION 1 — FSIS establishment choropleths (2017–2023) | map/html | `(name built from a variable: fig.write_html(html_path))` |
| `script2f-maps.py` | 145 | SECTION 1 — FSIS establishment choropleths (2017–2023) | figure (png) | `(name built from a variable: fig.write_image(png_path, width=1400, height=600, scale=2))` |
| `script2f-maps.py` | 258 | SECTION 2 — Mental health outcome value maps | map/html | `(name built from a variable: fig.write_html(html_path, include_plotlyjs="cdn"))` |
| `script2f-maps.py` | 261 | SECTION 2 — Mental health outcome value maps | figure (png) | `(name built from a variable: fig.write_image(png_path, width=320 * len(years), height=320)` |
| `script2f-maps.py` | 396 | SECTION 3 — Agricultural first-difference maps | map/html | `(name built from a variable: fig.write_html(html_path, include_plotlyjs="cdn"))` |
| `script2f-maps.py` | 399 | SECTION 3 — Agricultural first-difference maps | figure (png) | `(name built from a variable: fig.write_image(png_path, width=330 * len(years), height=340)` |
| `script2f-maps.py` | 406 | SECTION 3 — Agricultural first-difference maps | table (csv) | `(name built from a variable: pd.concat(summary_frames, ignore_index=True).to_csv(summary_)` |
| `script4s-species-overlap.py` | 61 |  | table (csv) | `{TODAY}_species_nan_as_zero.csv` |
| `script4s-species-overlap.py` | 77 |  | table (csv) | `{TODAY}_species_count_by_year.csv` |
| `script4s-species-overlap.py` | 92 |  | table (csv) | `{TODAY}_species_combination_census.csv` |
| `script4s-species-overlap.py` | 113 |  | table (csv) | `{TODAY}_species_pair_census.csv` |
| `script4s-species-overlap.py` | 136 |  | table (csv) | `{TODAY}_species_county_summary.csv` |
| `script4s-species-overlap.py` | 158 |  | table (csv) | `{TODAY}_species_cattle_check.csv` |
| `script4s-species-overlap.py` | 177 |  | figure | `(name built from a variable: fig.savefig(fp, dpi=150); plt.close(fig))` |
| `script4v-trend-plots.py` | 91 |  | figure | `{TODAY}_F1_means_by_group.png` |
| `script4v-trend-plots.py` | 92 |  | table (csv) | `{TODAY}_F1_means_by_group.csv` |
| `script4v-trend-plots.py` | 157 |  | figure | `{TODAY}_F2_event_time_{tag}.png` |
| `script4v-trend-plots.py` | 158 |  | table (csv) | `{TODAY}_F2_event_time_paths.csv` |
| `script4v-trend-plots.py` | 184 |  | table (csv) | `{TODAY}_F3_pre_window_slopes.csv` |
| `script4v-trend-plots.py` | 207 |  | figure | `{TODAY}_F3_pre_window_slope_difference.png` |
| `script4k-variable-register.py` | 414 |  | table (xlsx) | `(name built from a variable: README.to_excel(w,sheet_name="README",index=False))` |
| `script4k-variable-register.py` | 415 |  | table (xlsx) | `(name built from a variable: R.to_excel(w,sheet_name="variables",index=False))` |
| `script4k-variable-register.py` | 416 |  | table (xlsx) | `(name built from a variable: FS.to_excel(w,sheet_name="FSIS_panel",index=False))` |
| `script4k-variable-register.py` | 422 |  | table (csv) | `{TODAY}_VARIABLE_REGISTER.csv` |
