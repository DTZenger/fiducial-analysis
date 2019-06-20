import corner_histogram as ch
import unittest
import numpy as np

class Testhistos(unittest.TestCase):

    def test_of_angle_of_straight_line(self):
        x=np.ones(8)
        self.assertEqual(ch.get_val_diff_avg(x),0)

    def test_of_angle_of_ordered_numbers(self):
        x=np.arange(8)
        self.assertEqual(ch.get_val_diff_avg(x),-2)
    
    def test_of_column_of_ones(self):
        x=(np.ones(8)).T
        self.assertEqual(ch.get_val_diff_avg(x),0)
    
    def test_of_linear_relationship(self):
        x=np.array([8,16,8,0,24,64,56,16])
        self.assertEqual(ch.get_val_diff_avg(x),8)
    
        
    def test_the_data_reading(self):
        ph=ch.point_histogram("test1.csv")
        x=np.array([[1,1],[1,0],[0,0],[0,1]])
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                self.assertEqual(ph.data[i][j],x[i][j],msg='failed at index ({},{})'.format(i,j))
                
    def test_angle_measurements(self):
        ph=ch.point_histogram("test1.csv")

        self.assertEqual(0,ph.find_angle2())
        self.assertEqual(0,ph.find_angle(ph.data))
        
    def test_rotate_1_1_down_45_degs(self):
        x=ch.rotate_values(np.array([1,1]),np.pi/4)
        self.assertAlmostEqual(x[0],np.sqrt(2),delta=0.01)
        self.assertAlmostEqual(x[1],0,delta=0.01)
        
if __name__ == '__main__':
    unittest.main()