import unittest
import numpy as np
from io import StringIO
from contextlib import redirect_stdout
from tic_tac_toe import tic_tac_toe_Value_Iteration as ttt


class TestUtils(unittest.TestCase):
    def test_print_empty_board(self):
        board = np.zeros((3, 3), dtype=int)
        expected_output = (
            "  |   |  \n"
            "--+---+--\n"
            "  |   |  \n"
            "--+---+--\n"
            "  |   |  \n"
            "***\n"
        )
        with StringIO() as buf, redirect_stdout(buf):
            ttt._print_board(board)
            output = buf.getvalue()
        
        self.assertEqual(output, expected_output)

    def test_print_winner_board(self):
        board = np.array([[1, 1, 1],
                          [-1, 0, -1],
                          [0, 0, 0]])
        expected_output = (
            "X | X | X\n"
            "--+---+--\n"
            "O |   | O\n"
            "--+---+--\n"
            "  |   |  \n"
            "***\n"
        )
        with StringIO() as buf, redirect_stdout(buf):
            ttt._print_board(board)
            output = buf.getvalue()
        
        self.assertEqual(output, expected_output)

    def test_print_full_board(self):
        board = np.array([[1, -1, 1],
                          [-1, 1, -1],
                          [-1, 1, -1]])
        expected_output = (
            "X | O | X\n"
            "--+---+--\n"
            "O | X | O\n"
            "--+---+--\n"
            "O | X | O\n"
            "***\n"
        )
        with StringIO() as buf, redirect_stdout(buf):
            ttt._print_board(board)
            output = buf.getvalue()
        
        self.assertEqual(output, expected_output)

    def test_check_winner_empty_board(self):
        board = np.zeros((3, 3), dtype=int)
        winner = ttt._check_winner(board)
        self.assertIsNone(winner)

    def test_check_winner_non_terminated(self):
        board = np.array([[1, 0, 1],
                          [-1, 0, -1],
                          [0, 0, 0]])
        winner = ttt._check_winner(board)
        self.assertIsNone(winner)

    def test_check_winner_draw(self):
        board = np.array([[1, -1, 1],
                          [-1, 1, -1],
                          [-1, 1, -1]])
        winner = ttt._check_winner(board)
        expected_winner = ttt.PlayerSymbol.EMPTY
        self.assertEqual(winner, expected_winner)

    def test_check_winner_player_x_horizontal(self):
        board = np.array([[1, 1, 1],
                          [-1, 0, -1],
                          [0, 0, 0]])
        winner = ttt._check_winner(board)
        expected_winner = ttt.PlayerSymbol.X
        self.assertEqual(winner, expected_winner)

    def test_check_winner_player_x_vertical(self):
        board = np.array([[1, -1, 1],
                          [1, 0, -1],
                          [1, 0, 0]])
        winner = ttt._check_winner(board)
        expected_winner = ttt.PlayerSymbol.X
        self.assertEqual(winner, expected_winner)

    def test_check_winner_player_x_diagonal(self):
        board = np.array([[1, -1, 1],
                          [1, 1, -1],
                          [0, 0, 1]])
        winner = ttt._check_winner(board)
        expected_winner = ttt.PlayerSymbol.X
        self.assertEqual(winner, expected_winner)

    def test_check_winner_player_x_antidiagonal(self):
        board = np.array([[-1, -1, 1],
                          [1, 1, -1],
                          [1, 0, 1]])
        winner = ttt._check_winner(board)
        expected_winner = ttt.PlayerSymbol.X
        self.assertEqual(winner, expected_winner)

    def test_check_winner_player_y_horizontal(self):
        board = np.array([[1, 0, 1],
                          [-1, -1, -1],
                          [0, 0, 0]])
        winner = ttt._check_winner(board)
        expected_winner = ttt.PlayerSymbol.Y
        self.assertEqual(winner, expected_winner)
    
    def test_check_winner_player_y_vertical(self):
        board = np.array([[1, -1, 1],
                          [1, -1, -1],
                          [0, -1, 0]])
        winner = ttt._check_winner(board)
        expected_winner = ttt.PlayerSymbol.Y
        self.assertEqual(winner, expected_winner)

    def test_check_winner_player_y_diagonal(self):
        board = np.array([[-1, -1, 1],
                          [1, -1, -1],
                          [0, 1, -1]])
        winner = ttt._check_winner(board)
        expected_winner = ttt.PlayerSymbol.Y
        self.assertEqual(winner, expected_winner)

    def test_check_winner_player_y_antidiagonal(self):
        board = np.array([[1, -1, -1],
                          [1, -1, -1],
                          [-1, 1, 0]])
        winner = ttt._check_winner(board)
        expected_winner = ttt.PlayerSymbol.Y
        self.assertEqual(winner, expected_winner)

    def test_get_winner_name_x(self):
        winner = ttt._get_winner_name(ttt.PlayerSymbol.X)
        expected_name = "X"
        self.assertEqual(winner, expected_name)

    def test_get_winner_name_y(self):
        winner = ttt._get_winner_name(ttt.PlayerSymbol.Y)
        expected_name = "O"
        self.assertEqual(winner, expected_name)

    def test_get_winner_name_draw(self):
        winner = ttt._get_winner_name(ttt.PlayerSymbol.EMPTY)
        expected_name = "Draw"
        self.assertEqual(winner, expected_name)

    def test_available_positions_empty(self):
        board = np.zeros((3, 3), dtype=int)
        available_positions = ttt._get_available_positions(board)
        expected_positions = [(0, 0), (0, 1), (0, 2),
                              (1, 0), (1, 1), (1, 2),
                              (2, 0), (2, 1), (2, 2)]
        self.assertEqual(available_positions, expected_positions)

    def test_available_positions_corners(self):
        board = np.array([[0, -1, 0],
                          [1, 1, -1],
                          [0,-1, 0]])
        available_positions = ttt._get_available_positions(board)
        expected_positions = [(0, 0), (0, 2), (2, 0), (2, 2)]
        self.assertEqual(available_positions, expected_positions)

    def test_available_positions_stencil(self):
        board = np.array([[1, 0, 1],
                          [0, 0, 0],
                          [-1, 0, 1]])
        available_positions = ttt._get_available_positions(board)
        expected_positions = [(0, 1), (1, 0), (1, 1), (1, 2), (2, 1)]
        self.assertEqual(available_positions, expected_positions)
    
    def test_is_valid_empty_board(self):
        board = np.zeros((3, 3), dtype=int)
        valid = ttt._is_valid_board(board)
        expected = True
        self.assertEqual(valid, expected)

    def test_is_valid_too_large_board(self):
        board = np.zeros((4, 4), dtype=int)
        with self.assertRaises(AssertionError):
            ttt._is_valid_board(board)

    def test_is_valid_x_ahead_y_1(self):
        board = np.array([[0, 0, 0],
                          [0, 1, 0],
                          [0, 0, 0]])
        valid = ttt._is_valid_board(board)
        expected = True
        self.assertEqual(valid, expected)

    def test_is_valid_x_ahead_y_2(self):
        board = np.array([[0, 0, 0],
                          [0, 1, -1],
                          [0, 0, 1]])
        valid = ttt._is_valid_board(board)
        expected = True
        self.assertEqual(valid, expected)

    def test_is_valid_x_draw_y_1(self):
        board = np.array([[0, 0, 0],
                          [0, 1, -1],
                          [0, 0, 0]])
        valid = ttt._is_valid_board(board)
        expected = True
        self.assertEqual(valid, expected)

    def test_is_valid_x_draw_y_2(self):
        board = np.array([[1, -1, 1],
                          [-1, 1, -1],
                          [1, -1, 0]])
        valid = ttt._is_valid_board(board)
        expected = True
        self.assertEqual(valid, expected)

    def test_is_valid_x_winner(self):
        board = np.array([[1, -1, 0],
                          [0, 1, -1],
                          [0, 0, 1]])
        valid = ttt._is_valid_board(board)
        expected = True
        self.assertEqual(valid, expected)

    def test_is_valid_y_winner(self):
        board = np.array([[1, -1, -1],
                          [0, 1, -1],
                          [1, 1, -1]])
        valid = ttt._is_valid_board(board)
        expected = True
        self.assertEqual(valid, expected)

    def test_is_valid_draw(self):
        board = np.array([[1, -1, -1],
                          [-1, 1, 1],
                          [1, 1, -1]])
        valid = ttt._is_valid_board(board)
        expected = True
        self.assertEqual(valid, expected)

    def test_is_valid_wrong_symbols(self):
        board = np.array([[1, -1, -1],
                          [2, 1, -1],
                          [1, 1, -1]])
        with self.assertRaises(AssertionError):
            ttt._is_valid_board(board)

    def test_is_invalid_too_much_x_1(self):
        board = np.array([[1, 1, 1],
                          [1, 1, 1],
                          [1, 1, 1]])
        valid = ttt._is_valid_board(board)
        expected = False
        self.assertEqual(valid, expected)

    def test_is_invalid_too_much_x_2(self):
        board = np.array([[1, 1, 0],
                          [0, 0, 0],
                          [0, 0, 0]])
        valid = ttt._is_valid_board(board)
        expected = False
        self.assertEqual(valid, expected)

    def test_is_invalid_too_much_y_1(self):
        board = np.array([[-1, -1, 1],
                          [-1, -1, 1],
                          [-1, -1, 1]])
        valid = ttt._is_valid_board(board)
        expected = False
        self.assertEqual(valid, expected)

    def test_is_invalid_too_much_y_2(self):
        board = np.array([[-1, -1, 0],
                          [0, 0, 0],
                          [0, 0, 1]])
        valid = ttt._is_valid_board(board)
        expected = False
        self.assertEqual(valid, expected)

    def test_generate_all_states(self):
        all_states = ttt._generate_all_states()


if __name__ == "__main__":
    unittest.main()