-- Analytical SQL over the BigMart sales table (SQLite). Run: python scripts/sql_analytics.py
-- Table `sales` is loaded from Train.csv by the script; sales values are yearly sales per item-outlet pair.

-- name: data_quality_profile
-- Defects found before modelling: missing values, inconsistent labels, impossible zeros.
SELECT COUNT(*)                                                        AS rows_total,
       SUM(Item_Weight IS NULL)                                        AS missing_weight,
       SUM(Outlet_Size IS NULL)                                        AS missing_outlet_size,
       SUM(Item_Visibility = 0)                                        AS zero_visibility,
       SUM(Item_Fat_Content IN ('LF','low fat','reg'))                 AS inconsistent_fat_labels,
       COUNT(DISTINCT Item_Identifier)                                 AS items,
       COUNT(DISTINCT Outlet_Identifier)                               AS outlets
FROM sales;

-- name: revenue_share_by_outlet_type
-- Window function over a grouped result: each outlet type's share of total sales.
SELECT Outlet_Type,
       COUNT(*)                                                         AS rows_,
       ROUND(SUM(Item_Outlet_Sales))                                    AS revenue,
       ROUND(100.0 * SUM(Item_Outlet_Sales) / SUM(SUM(Item_Outlet_Sales)) OVER (), 1) AS pct_of_total,
       ROUND(AVG(Item_Outlet_Sales))                                    AS avg_sales_per_row
FROM sales GROUP BY Outlet_Type ORDER BY revenue DESC;

-- name: top3_item_types_per_outlet_type
-- ROW_NUMBER() partitioned ranking: the three biggest categories inside each outlet type.
WITH t AS (
  SELECT Outlet_Type, Item_Type, ROUND(SUM(Item_Outlet_Sales)) AS revenue,
         ROW_NUMBER() OVER (PARTITION BY Outlet_Type ORDER BY SUM(Item_Outlet_Sales) DESC) AS rnk
  FROM sales GROUP BY Outlet_Type, Item_Type)
SELECT Outlet_Type, rnk, Item_Type, revenue FROM t WHERE rnk <= 3 ORDER BY Outlet_Type, rnk;

-- name: revenue_concentration_pareto
-- Cumulative window: how many items does it take to reach 50% / 80% of all sales?
WITH item AS (
  SELECT Item_Identifier, SUM(Item_Outlet_Sales) AS revenue FROM sales GROUP BY Item_Identifier),
ranked AS (
  SELECT revenue, ROW_NUMBER() OVER (ORDER BY revenue DESC) AS n,
         SUM(revenue) OVER (ORDER BY revenue DESC ROWS UNBOUNDED PRECEDING) * 1.0 / SUM(revenue) OVER () AS cum_share
  FROM item)
SELECT 'items for 50% of sales' AS metric, MIN(n) AS items, ROUND(100.0 * MIN(n) / (SELECT COUNT(*) FROM item), 1) AS pct_of_catalogue FROM ranked WHERE cum_share >= 0.5
UNION ALL
SELECT 'items for 80% of sales', MIN(n), ROUND(100.0 * MIN(n) / (SELECT COUNT(*) FROM item), 1) FROM ranked WHERE cum_share >= 0.8;

-- name: price_quartile_effect
-- NTILE(4) on price: does a higher-priced item sell more per row, and who carries the revenue?
WITH q AS (SELECT *, NTILE(4) OVER (ORDER BY Item_MRP) AS price_quartile FROM sales)
SELECT price_quartile, ROUND(MIN(Item_MRP)) AS mrp_from, ROUND(MAX(Item_MRP)) AS mrp_to,
       ROUND(AVG(Item_Outlet_Sales)) AS avg_sales_per_row,
       ROUND(100.0 * SUM(Item_Outlet_Sales) / SUM(SUM(Item_Outlet_Sales)) OVER (), 1) AS pct_of_revenue
FROM q GROUP BY price_quartile ORDER BY price_quartile;

-- name: outlet_vs_outlet_type_benchmark
-- Each outlet against the average of its own type (same window, partitioned): a store-performance scorecard.
WITH o AS (
  SELECT Outlet_Identifier, Outlet_Type, Outlet_Establishment_Year AS opened, ROUND(AVG(Item_Outlet_Sales)) AS avg_sales
  FROM sales GROUP BY Outlet_Identifier, Outlet_Type, Outlet_Establishment_Year)
SELECT Outlet_Identifier, Outlet_Type, opened, avg_sales,
       ROUND(AVG(avg_sales) OVER (PARTITION BY Outlet_Type)) AS type_avg,
       ROUND(100.0 * avg_sales / AVG(avg_sales) OVER (PARTITION BY Outlet_Type) - 100, 1) AS pct_vs_type_avg
FROM o ORDER BY Outlet_Type, avg_sales DESC;
