// Single processing element for an output-stationary INT8 systolic array.
// Multiplies a_in * b_in, accumulates into acc, and forwards a_in / b_in.
// Intended as the building block for a parameterized NxN mesh.

`default_nettype none

module systolic_pe #(
    parameter int A_WIDTH = 8,
    parameter int B_WIDTH = 8,
    parameter int ACC_WIDTH = 32
) (
    input  logic                     clk,
    input  logic                     rst_n,
    input  logic                     en,
    input  logic                     clear,
    input  logic signed [A_WIDTH-1:0] a_in,
    input  logic signed [B_WIDTH-1:0] b_in,
    output logic signed [A_WIDTH-1:0] a_out,
    output logic signed [B_WIDTH-1:0] b_out,
    output logic signed [ACC_WIDTH-1:0] acc
);

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            a_out <= '0;
            b_out <= '0;
            acc   <= '0;
        end else if (clear) begin
            acc <= '0;
        end else if (en) begin
            a_out <= a_in;
            b_out <= b_in;
            acc   <= acc + (ACC_WIDTH'(a_in) * ACC_WIDTH'(b_in));
        end
    end

endmodule
