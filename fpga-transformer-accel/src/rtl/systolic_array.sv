// Parameterized NxN output-stationary systolic array.
// Weights flow down each column, activations flow across each row,
// accumulators are read out after clearing between tiles.

`default_nettype none

module systolic_array #(
    parameter int N = 8,
    parameter int A_WIDTH = 8,
    parameter int B_WIDTH = 8,
    parameter int ACC_WIDTH = 32
) (
    input  logic                        clk,
    input  logic                        rst_n,
    input  logic                        en,
    input  logic                        clear,
    input  logic signed [A_WIDTH-1:0]   a_row [N],
    input  logic signed [B_WIDTH-1:0]   b_col [N],
    output logic signed [ACC_WIDTH-1:0] acc_out [N][N]
);

    logic signed [A_WIDTH-1:0] a_h [N][N+1];
    logic signed [B_WIDTH-1:0] b_v [N+1][N];

    for (genvar r = 0; r < N; r++) assign a_h[r][0] = a_row[r];
    for (genvar c = 0; c < N; c++) assign b_v[0][c] = b_col[c];

    for (genvar r = 0; r < N; r++) begin : gen_row
        for (genvar c = 0; c < N; c++) begin : gen_col
            systolic_pe #(
                .A_WIDTH(A_WIDTH),
                .B_WIDTH(B_WIDTH),
                .ACC_WIDTH(ACC_WIDTH)
            ) pe (
                .clk(clk),
                .rst_n(rst_n),
                .en(en),
                .clear(clear),
                .a_in(a_h[r][c]),
                .b_in(b_v[r][c]),
                .a_out(a_h[r][c+1]),
                .b_out(b_v[r+1][c]),
                .acc(acc_out[r][c])
            );
        end
    end

endmodule
