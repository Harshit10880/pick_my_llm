import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

def create_radar_chart(ranked_df: pd.DataFrame) -> go.Figure:
    """
    Builds an interactive Plotly Radar / Spider Chart comparing models across
    Accuracy, Normalized Speed, Cost Efficiency, and Final Score.
    """
    categories = ['Accuracy (%)', 'Speed Score', 'Cost Efficiency', 'Overall Score']
    
    fig = go.Figure()

    # Pre-defined vibrant color palette
    colors = ['#38bdf8', '#f43f5e', '#a855f7', '#22c55e', '#eab308', '#f97316']

    for idx, (_, row) in enumerate(ranked_df.iterrows()):
        model_name = row["model"]
        acc = float(row.get("accuracy_pct", 0))
        speed_score = float(row.get("norm_speed", 0)) * 100
        cost_score = float(row.get("norm_cost", 0)) * 100
        final_score = float(row.get("final_score", 0))
        
        values = [acc, speed_score, cost_score, final_score]
        # Close the radar loop by repeating the first value
        values_closed = values + [values[0]]
        cats_closed = categories + [categories[0]]
        
        color = colors[idx % len(colors)]
        
        fig.add_trace(go.Scatterpolar(
            r=values_closed,
            theta=cats_closed,
            fill='toself',
            name=model_name,
            line=dict(color=color, width=2),
            opacity=0.65
        ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                showticklabels=True,
                gridcolor='#334155'
            ),
            angularaxis=dict(
                gridcolor='#334155'
            ),
            bgcolor='rgba(15, 23, 42, 0.4)'
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#e2e8f0', size=12),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.25,
            xanchor="center",
            x=0.5
        ),
        margin=dict(l=40, r=40, t=30, b=50),
        height=420
    )
    return fig

def create_category_chart(results_df: pd.DataFrame) -> go.Figure:
    """
    Builds a grouped bar chart showing accuracy percentage broken down by question category.
    """
    if "category" not in results_df.columns or "is_correct" not in results_df.columns:
        return None
        
    cat_df = results_df.groupby(["category", "model"])["is_correct"].agg(
        total="count",
        correct="sum"
    ).reset_index()
    
    cat_df["accuracy_pct"] = ((cat_df["correct"] / cat_df["total"]) * 100).round(1)
    
    fig = px.bar(
        cat_df,
        x="category",
        y="accuracy_pct",
        color="model",
        barmode="group",
        labels={"accuracy_pct": "Accuracy (%)", "category": "Topic Category", "model": "Model"},
        color_discrete_sequence=['#38bdf8', '#f43f5e', '#a855f7', '#22c55e', '#eab308'],
        text="accuracy_pct"
    )
    
    fig.update_traces(texttemplate='%{text}%', textposition='outside')
    fig.update_layout(
        yaxis=dict(range=[0, 115], gridcolor='#334155'),
        xaxis=dict(gridcolor='#334155'),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#e2e8f0'),
        legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5),
        margin=dict(l=20, r=20, t=20, b=40),
        height=380
    )
    return fig

def create_tradeoff_chart(ranked_df: pd.DataFrame) -> go.Figure:
    """
    Creates a 2D Speed (Latency) vs Accuracy trade-off scatter plot.
    Top-left quadrant = fastest & most accurate (Pareto optimal).
    """
    fig = px.scatter(
        ranked_df,
        x="avg_latency",
        y="accuracy_pct",
        size="final_score",
        color="model",
        text="model",
        labels={"avg_latency": "Avg Latency (Seconds - Lower is better)", "accuracy_pct": "Accuracy (%) - Higher is better"},
        color_discrete_sequence=['#38bdf8', '#f43f5e', '#a855f7', '#22c55e'],
        size_max=30
    )
    
    fig.update_traces(textposition='top center', marker=dict(opacity=0.85, line=dict(width=1, color='white')))
    fig.update_layout(
        yaxis=dict(range=[0, 110], gridcolor='#334155'),
        xaxis=dict(gridcolor='#334155'),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#e2e8f0'),
        margin=dict(l=20, r=20, t=20, b=40),
        height=380,
        showlegend=False
    )
    return fig
