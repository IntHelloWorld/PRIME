class Car extends Vehicle {
    private String brand;
    private int year;

    public Car(String brand, int year) {
        this.brand = brand;
        this.year = year;
    }

    @Override
    public void start() {
        System.out.println("汽车发动");
    }

    @Override
    public void stop() {
        System.out.println("汽车停止");
    }

    @Override
    public String getInfo() {
        return "品牌: " + brand + ", 年份: " + year;
    }
}
