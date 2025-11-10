# Biochemical Pathway Diagram in R
# System of differential equations pathway visualization

# Install required packages if needed
# install.packages("DiagrammeR")
# install.packages("igraph")

library(DiagrammeR)

# Method 1: Using DiagrammeR with Graphviz DOT notation
pathway_diagram <- grViz("
digraph pathway {
  
  # Graph attributes
  graph [layout = neato, 
         overlap = false, 
         fontsize = 16,
         label = 'Biochemical Pathway System',
         labelloc = 't']
  
  # Node definitions
  node [shape = circle, 
        style = filled, 
        fillcolor = lightblue, 
        fontsize = 18,
        width = 1.2,
        fixedsize = true]
  
  X1 [pos = '2,4!', label = 'X₁']
  X2 [pos = '6,4!', label = 'X₂']
  X3 [pos = '4,2!', label = 'X₃']
  X4 [pos = '2,0!', label = 'X₄', fillcolor = lightcoral]
  
  # Production edges (solid blue arrows)
  edge [color = blue, 
        penwidth = 2.5,
        arrowsize = 1.2]
  
  X3 -> X1 [label = <<i>α</i><sub>1</sub>X<sub>3</sub><sup>g<sub>13</sub></sup>>]
  X1 -> X2 [label = <<i>α</i><sub>2</sub>X<sub>1</sub><sup>g<sub>21</sub></sup>>]
  X2 -> X3 [label = <<i>α</i><sub>3</sub>X<sub>2</sub><sup>g<sub>32</sub></sup>>]
  X1 -> X4 [label = <<i>α</i><sub>4</sub>X<sub>1</sub><sup>g<sub>41</sub></sup>>]
  
  # Inhibition edge (dashed red)
  edge [color = red, 
        style = dashed, 
        arrowhead = tee,
        penwidth = 2.5]
  
  X4 -> X3 [label = <<i>β</i><sub>3</sub>X<sub>4</sub><sup>h<sub>34</sub></sup>>]
  
  # Self-degradation loops (gray dotted)
  edge [color = gray, 
        style = dotted, 
        penwidth = 2,
        dir = both,
        arrowtail = none,
        arrowhead = none]
  
  X1 -> X1 [label = <<i>β</i><sub>1</sub>X<sub>1</sub><sup>h<sub>11</sub></sup>>]
  X2 -> X2 [label = <<i>β</i><sub>2</sub>X<sub>2</sub><sup>h<sub>22</sub></sup>>]
  X3 -> X3 [label = <<i>β</i><sub>3</sub>X<sub>3</sub><sup>h<sub>33</sub></sup>>]
  X4 -> X4 [label = <<i>β</i><sub>4</sub>X<sub>4</sub><sup>h<sub>44</sub></sup>>]
}
")

# Display the diagram
pathway_diagram

# Save as PDF
# pathway_diagram %>% export_svg() %>% charToRaw() %>% rsvg::rsvg_pdf("pathway_diagram.pdf")


# ============================================================================
# Method 2: Using igraph for network visualization
# ============================================================================

library(igraph)

# Create edge list
edges <- data.frame(
  from = c("X3", "X1", "X2", "X1", "X4", "X1", "X2", "X3", "X4"),
  to = c("X1", "X2", "X3", "X4", "X3", "X1", "X2", "X3", "X4"),
  type = c("production", "production", "production", "production", 
           "inhibition", "degradation", "degradation", "degradation", "degradation"),
  label = c("α₁X₃^g₁₃", "α₂X₁^g₂₁", "α₃X₂^g₃₂", "α₄X₁^g₄₁",
            "β₃X₄^h₃₄", "β₁X₁^h₁₁", "β₂X₂^h₂₂", "β₃X₃^h₃₃", "β₄X₄^h₄₄")
)

# Create graph
g <- graph_from_data_frame(edges, directed = TRUE)

# Set visual attributes
V(g)$color <- c("lightblue", "lightblue", "lightblue", "lightcoral")
V(g)$size <- 40
V(g)$label.cex <- 1.5
V(g)$label.color <- "black"

# Edge colors based on type
E(g)$color <- ifelse(edges$type == "production", "blue",
                     ifelse(edges$type == "inhibition", "red", "gray"))
E(g)$width <- ifelse(edges$type == "degradation", 1.5, 2.5)
E(g)$lty <- ifelse(edges$type == "inhibition", 2,
                   ifelse(edges$type == "degradation", 3, 1))

# Set layout
layout_coords <- matrix(c(
  2, 4,   # X1
  6, 4,   # X2
  4, 2,   # X3
  2, 0    # X4
), ncol = 2, byrow = TRUE)

# Plot
par(mar = c(1, 1, 3, 1))
plot(g, 
     layout = layout_coords,
     edge.arrow.size = 0.8,
     edge.curved = 0.2,
     edge.label = edges$label,
     edge.label.cex = 0.7,
     main = "Biochemical Pathway System")

# Add legend
legend("bottomright", 
       legend = c("Production", "Inhibition", "Degradation"),
       col = c("blue", "red", "gray"),
       lty = c(1, 2, 3),
       lwd = 2,
       cex = 0.8)


# ============================================================================
# Method 3: Simple base R plotting with arrows
# ============================================================================

plot_pathway_base <- function() {
  # Set up plot
  par(mar = c(1, 1, 3, 1))
  plot(NULL, xlim = c(0, 8), ylim = c(-1, 5), 
       xlab = "", ylab = "", axes = FALSE,
       main = "Biochemical Pathway System", cex.main = 1.5)
  
  # Node positions
  nodes <- data.frame(
    name = c("X₁", "X₂", "X₃", "X₄"),
    x = c(2, 6, 4, 2),
    y = c(4, 4, 2, 0),
    color = c("lightblue", "lightblue", "lightblue", "lightcoral")
  )
  
  # Draw nodes
  for(i in 1:nrow(nodes)) {
    symbols(nodes$x[i], nodes$y[i], circles = 0.4, 
            inches = FALSE, add = TRUE, 
            bg = nodes$color[i], fg = "black", lwd = 2)
    text(nodes$x[i], nodes$y[i], nodes$name[i], cex = 1.5, font = 2)
  }
  
  # Draw arrows
  # X3 -> X1
  arrows(3.6, 2.3, 2.3, 3.7, col = "blue", lwd = 2.5, length = 0.15)
  text(2.8, 3.2, expression(alpha[1]*X[3]^g[13]), cex = 0.8)
  
  # X1 -> X2
  arrows(2.4, 4, 5.6, 4, col = "blue", lwd = 2.5, length = 0.15)
  text(4, 4.4, expression(alpha[2]*X[1]^g[21]), cex = 0.8)
  
  # X2 -> X3
  arrows(5.7, 3.7, 4.3, 2.3, col = "blue", lwd = 2.5, length = 0.15)
  text(5.2, 3.2, expression(alpha[3]*X[2]^g[32]), cex = 0.8)
  
  # X1 -> X4
  arrows(2, 3.6, 2, 0.4, col = "blue", lwd = 2.5, length = 0.15)
  text(1.3, 2, expression(alpha[4]*X[1]^g[41]), cex = 0.8)
  
  # X4 inhibits X3 (dashed red)
  arrows(2.3, 0.3, 3.7, 1.7, col = "red", lwd = 2.5, 
         length = 0.15, lty = 2)
  text(3.2, 0.8, expression(beta[3]*X[4]^h[34]), cex = 0.8, col = "red")
  
  # Self-degradation loops (shown as circular arrows)
  # X1
  draw.circle(1.4, 4, 0.3, nv = 30, border = "gray", lty = 3, lwd = 1.5)
  text(0.8, 4, expression(beta[1]*X[1]^h[11]), cex = 0.7, col = "gray40")
  
  # X2
  draw.circle(6.6, 4, 0.3, nv = 30, border = "gray", lty = 3, lwd = 1.5)
  text(7.2, 4, expression(beta[2]*X[2]^h[22]), cex = 0.7, col = "gray40")
  
  # X3
  draw.circle(4.6, 2, 0.3, nv = 30, border = "gray", lty = 3, lwd = 1.5)
  text(5.2, 2, expression(beta[3]*X[3]^h[33]), cex = 0.7, col = "gray40")
  
  # X4
  draw.circle(1.4, 0, 0.3, nv = 30, border = "gray", lty = 3, lwd = 1.5)
  text(0.8, 0, expression(beta[4]*X[4]^h[44]), cex = 0.7, col = "gray40")
  
  # Legend
  legend("bottom", 
         legend = c("Production", "Inhibition", "Degradation"),
         col = c("blue", "red", "gray"),
         lty = c(1, 2, 3),
         lwd = 2,
         cex = 0.8,
         horiz = TRUE)
}

# Helper function for drawing circles (degradation loops)
draw.circle <- function(x, y, r, nv = 100, border = "black", lty = 1, lwd = 1) {
  angle <- seq(0, 2 * pi, length.out = nv)
  lines(x + r * cos(angle), y + r * sin(angle), 
        col = border, lty = lty, lwd = lwd)
}

# Call the function
plot_pathway_base()


# ============================================================================
# Bonus: System dynamics simulation (optional)
# ============================================================================

# Define the ODE system
library(deSolve)

pathway_ode <- function(t, state, parameters) {
  with(as.list(c(state, parameters)), {
    dX1 <- alpha1 * X3^g13 - beta1 * X1^h11
    dX2 <- alpha2 * X1^g21 - beta2 * X2^h22
    dX3 <- alpha3 * X2^g32 - beta3 * X3^h33 * X4^h34
    dX4 <- alpha4 * X1^g41 - beta4 * X4^h44
    
    list(c(dX1, dX2, dX3, dX4))
  })
}

# Example parameters (adjust as needed)
parameters <- c(
  alpha1 = 1.0, alpha2 = 1.0, alpha3 = 1.0, alpha4 = 0.5,
  beta1 = 0.5, beta2 = 0.5, beta3 = 0.5, beta4 = 0.5,
  g13 = 2, g21 = 2, g32 = 2, g41 = 2,
  h11 = 1, h22 = 1, h33 = 1, h34 = 2, h44 = 1
)

# Initial conditions
state <- c(X1 = 1, X2 = 1, X3 = 1, X4 = 0.5)

# Time span
times <- seq(0, 50, by = 0.1)

# Solve ODE
out <- ode(y = state, times = times, func = pathway_ode, parms = parameters)

# Plot time series
par(mfrow = c(2, 2), mar = c(4, 4, 2, 1))
plot(out[, 1], out[, 2], type = "l", col = "blue", lwd = 2,
     xlab = "Time", ylab = "X₁", main = "X₁ Dynamics")
plot(out[, 1], out[, 3], type = "l", col = "green", lwd = 2,
     xlab = "Time", ylab = "X₂", main = "X₂ Dynamics")
plot(out[, 1], out[, 4], type = "l", col = "purple", lwd = 2,
     xlab = "Time", ylab = "X₃", main = "X₃ Dynamics")
plot(out[, 1], out[, 5], type = "l", col = "red", lwd = 2,
     xlab = "Time", ylab = "X₄", main = "X₄ Dynamics")

