namespace ImageViewer
{
    public partial class Form1 : Form
    {
        Bitmap source_image;
        Bitmap target_image;
        public Form1()
        {
            InitializeComponent();
        }

        private void openToolStripMenuItem_Click(object sender, EventArgs e)
        {
            OpenFileDialog openFileDialog = new OpenFileDialog();
            openFileDialog.ShowDialog();
            if(openFileDialog.FileName != null)
            {
                source_image = new Bitmap(openFileDialog.FileName);
            }
            pictureBox1.Image = source_image;
            pictureBox1.Refresh();
        }

        private void saveToolStripMenuItem_Click(object sender, EventArgs e)
        {
            SaveFileDialog saveFileDialog = new SaveFileDialog();
            saveFileDialog.ShowDialog();
            if(saveFileDialog.FileName != null)
            {
              
                target_image.Save(saveFileDialog.FileName);
            }
        }

        private void inverteToolStripMenuItem_Click(object sender, EventArgs e)
        {
            target_image = new Bitmap(source_image);

            for(int i = 0; i < target_image.Width; i++)
                for (int j = 0; j < target_image.Height; j++)
                {
                    Color pxl= target_image.GetPixel(i, j);
                    target_image.SetPixel(i, j, Color.FromArgb(255 - pxl.R, 255 - pxl.G, 255 - pxl.B));

                }
            pictureBox1.Image = target_image;
            pictureBox1.Refresh();


        }

        private void Form1_Load(object sender, EventArgs e)
        {

        }
    }
}